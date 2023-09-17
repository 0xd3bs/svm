use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorAdd, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{
        HALF, ONE, FP16x16Impl, FP16x16Div, FP16x16Print, FP16x16IntoI32
    }
};

use svm::{help::less};


fn from_u32_to_fixedtype(from_u32: @Tensor<u32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut from_u32_data = *from_u32.data;
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    loop {
        match from_u32_data.pop_front() {
            Option::Some(item) => {
                data_result.append(FixedType { mag: *item, sign: false });
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TensorTrait::<FixedType>::new(*from_u32.shape, data_result.span(), Option::Some(extra))
}

fn calculate_loss(
    w: Tensor<FixedType>,
    x_train: Tensor<FixedType>,
    y_train: Tensor<FixedType>,
    c: Tensor<FixedType>,
    one_tensor: @Tensor<FixedType>,
    half_tensor: @Tensor<FixedType>
) -> FixedType {
    let tensor_size = FP16x16Impl::new_unscaled(y_train.data.len(), false);

    let pre_cumsum = *one_tensor - y_train * x_train.matmul(@w);
    let cumsum = pre_cumsum.cumsum(0, Option::None(()), Option::None(()));
    let sum = cumsum.data[pre_cumsum.data.len() - 1];
    let mean = FP16x16Div::div(*sum, tensor_size);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let mean_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![mean].span(), extra: Option::Some(extra),
    );

    let regularization_term = *half_tensor * (w.matmul(@w));
    let loss_tensor = mean_tensor + c * regularization_term;

    loss_tensor.at(array![0].span())
}

fn calculate_gradient(
    w: Tensor<FixedType>,
    x_train: Tensor<FixedType>,
    y_train: Tensor<FixedType>,
    c: Tensor<FixedType>,
    one_tensor: @Tensor<FixedType>,
    neg_one_tensor: @Tensor<FixedType>
) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor_size = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(y_train.data.len(), false)].span(),
        extra: Option::Some(extra),
    );

    let mask = (y_train * x_train.matmul(@w)).less(one_tensor);
    //let mask = (y_train * x_train.matmul(@w));
    //let mask = less(*one_tensor, @mask);
    //let mask = less(@mask, *one_tensor);
    let mask = from_u32_to_fixedtype(@mask);

    let gradient = (((mask * y_train).matmul(@x_train) / tensor_size) * *neg_one_tensor) + (c * w);

    gradient
}

fn train_step(
    x: Tensor<FixedType>,
    y: Tensor<FixedType>,
    ref w: Tensor<FixedType>,
    learning_rate: FixedType,
    one_tensor: @Tensor<FixedType>,
    half_tensor: @Tensor<FixedType>,
    neg_one_tensor: @Tensor<FixedType>
) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let learning_rate_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![learning_rate].span(), extra: Option::Some(extra),
    );

    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let gradient = calculate_gradient(w, x, y, c, one_tensor, neg_one_tensor);
    w = w - (learning_rate_tensor * gradient);

    w
}

fn train(
    x: Tensor<FixedType>,
    y: Tensor<FixedType>,
    init_w: Tensor<FixedType>,
    learning_rate: FixedType,
    epoch: u32
) -> (Tensor<FixedType>, FixedType, FixedType) {
    let mut i = 1_u32;
    let mut w = init_w;

    'LOOPING...'.print();
    epoch.print();

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let half_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(HALF, false)].span(),
        extra: Option::Some(extra),
    );

    let neg_one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(ONE, true)].span(),
        extra: Option::Some(extra),
    );

    let mut initial_loss = FP16x16Impl::ZERO();
    let mut final_loss = FP16x16Impl::ZERO();

    loop {
        if i > epoch {
            break ();
        }

        w = train_step(x, y, ref w, learning_rate, @one_tensor, @half_tensor, @neg_one_tensor);
        if i == 1 {
            initial_loss = calculate_loss(w, x, y, c, @one_tensor, @half_tensor);
        } else if i == epoch {
            final_loss = calculate_loss(w, x, y, c, @one_tensor, @half_tensor);
        };
        i += 1;
    };

    (w, initial_loss, final_loss)
}
