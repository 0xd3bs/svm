use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams, unravel_index},
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

fn convert(mut from_to: Tensor<u32>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    loop {
        match from_to.data.pop_front() {
            Option::Some(item) => {
                data_result.append(FixedType { mag: *item, sign: false });
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(from_to.shape, data_result.span(), Option::Some(extra));
}

fn calculate_loss(
    w: Tensor<FixedType>,
    x_train: Tensor<FixedType>,
    y_train: Tensor<FixedType>,
    c: Tensor<FixedType>
) -> FixedType {
    let tensor_size = FP16x16Impl::new_unscaled(y_train.data.len(), false);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
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

    let pre_cumsum = one_tensor - y_train * x_train.matmul(@w);
    let cumsum = pre_cumsum.cumsum(0, Option::None(()), Option::None(()));
    let sum = cumsum.data[pre_cumsum.data.len() - 1];
    let mut mean = FP16x16Div::div(*sum, tensor_size);

    let mean_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![mean].span(), extra: Option::Some(extra),
    );

    let regularization_term = half_tensor * (w.matmul(@w));
    let loss_tensor = mean_tensor + c * regularization_term;

    loss_tensor.at(array![0].span())
}

fn calculate_gradient(
    w: Tensor<FixedType>,
    x_train: Tensor<FixedType>,
    y_train: Tensor<FixedType>,
    c: Tensor<FixedType>
) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let mut tensor_size = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(y_train.data.len(), false)].span(),
        extra: Option::Some(extra),
    );

    let one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    let neg_one_tensor = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(ONE, true)].span(),
        extra: Option::Some(extra),
    );

    let mask = (y_train * x_train.matmul(@w)).less(@one_tensor);

    let mask_convert = convert(mask);

    let pre_gradient = (mask_convert * y_train).matmul(@x_train);

    let mut gradient = ((pre_gradient / tensor_size) * neg_one_tensor) + (c * w);

    gradient
}

fn train_step(
    x: Tensor<FixedType>, y: Tensor<FixedType>, ref w: Tensor<FixedType>, learning_rate: FixedType
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

    let gradient = calculate_gradient(w, x, y, c);
    w = w - (learning_rate_tensor * gradient);

    w
}

fn train(
    x: Tensor<FixedType>,
    y: Tensor<FixedType>,
    init_w: Tensor<FixedType>,
    learning_rate: FixedType,
    epoch: u32
) -> (Tensor<FixedType>, Array::<FixedType>) {
    let mut i = 0_u32;
    let mut w = init_w;

    let mut loss_values = ArrayTrait::<FixedType>::new();

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let c = TensorTrait::new(
        shape: array![1].span(),
        data: array![FP16x16Impl::ONE()].span(),
        extra: Option::Some(extra),
    );

    loop {
        if i >= epoch {
            break ();
        }

        w = train_step(x, y, ref w, learning_rate);
        loss_values.append(calculate_loss(w, x, y, c));
        i += 1;
    };

    (w, loss_values)
}
