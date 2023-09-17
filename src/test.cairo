use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{FP16x16Impl, FP16x16Div, FP16x16PartialOrd, FP16x16Print}
};

use svm::{
    generated::{X_train::X_train, Y_train::Y_train, X_test::X_test, Y_test::Y_test}, train::{train}
};

#[test]
#[available_gas(99999999999999999)]
fn test() {
    let x_train = X_train();
    let x_test = X_test();
    let y_train = Y_train();
    let y_test = Y_test();

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    let feature_size = *x_train.shape[1];

    let mut zero_array = ArrayTrait::new();

    let mut i = 0_u32;
    loop {
        if i >= feature_size {
            break ();
        }
        zero_array.append(FP16x16Impl::ZERO());
        i += 1;
    };

    let initial_w = TensorTrait::new(
        shape: array![feature_size].span(), data: zero_array.span(), extra: Option::Some(extra),
    );

    let learning_rate = FixedTrait::new(655, false); // 655 is 0.01

    let (final_w, initial_loss, final_loss) = train(
        x_train, y_train, initial_w, learning_rate, 100_u32
    );

    //'final_w.0'.print();
    //(*final_w.data.at(0)).print();
    //'final_w.1'.print();
    //(*final_w.data.at(1)).print();
    //'final_w.2'.print();
    //(*final_w.data.at(2)).print();

    '(initial_loss)'.print();
    (initial_loss).print();
    '(final_loss)'.print();
    (final_loss).print();

    //let final_y_pred = pred(x_test, final_w);

    assert(final_loss < initial_loss, 'no decrease in training loss');
}
