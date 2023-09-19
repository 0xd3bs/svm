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

use svm::{ml_math::{calculate_loss, calculate_gradient}};

// Performs a training step for each iteration during model training
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

    //'gradient.shape.len()'.print();
    //gradient.shape.len().print();
    //'learning_rate_shape.len()'.print();
    //learning_rate_tensor.shape.len().print();
    //'w.len()'.print();
    //w.shape.len().print();

    w = w - (learning_rate_tensor * gradient);

    w
}

// Trains the machine learning model.
fn train(
    x: Tensor<FixedType>,
    y: Tensor<FixedType>,
    init_w: Tensor<FixedType>,
    learning_rate: FixedType,
    epoch: u32
) -> (Tensor<FixedType>, FixedType, FixedType) {
    let mut i = 1_u32;
    let mut otro_w = init_w;

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

    let mut initial_loss = FixedTrait::ZERO();
    let mut final_loss = FixedTrait::ZERO();

    if epoch > 0_u32 {
        //'FIRST'.print();
        initial_loss = calculate_loss(otro_w, x, y, c, @one_tensor, @half_tensor);
    };

    loop {
        if i > epoch {
            //'break'.print();
            //'i'.print();
            i.print();
            break ();
        }

        let partial_loss = calculate_loss(otro_w, x, y, c, @one_tensor, @half_tensor);

        //i.print();
        //(partial_loss).print();

        otro_w =
            train_step(x, y, ref otro_w, learning_rate, @one_tensor, @half_tensor, @neg_one_tensor);
        i += 1;
    };

    final_loss = calculate_loss(otro_w, x, y, c, @one_tensor, @half_tensor);

    (otro_w, initial_loss, final_loss)
}
