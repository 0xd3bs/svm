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

use svm::{help::{less, from_u32_to_fixedtype, _less}};

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
        data: array![FixedTrait::new_unscaled(y_train.data.len(), false)].span(),
        extra: Option::Some(extra),
    );

    //let mask = (y_train * x_train.matmul(@w)).less(one_tensor);
    //let mask = from_u32_to_fixedtype(@mask, @y_train);

    let mask = (y_train * x_train.matmul(@w));
    let mask = less(@mask, one_tensor);

    let gradient = (((mask * y_train).matmul(@x_train) / tensor_size) * *neg_one_tensor) + (c * w);

    gradient
}