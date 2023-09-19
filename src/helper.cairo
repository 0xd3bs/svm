use debug::PrintTrait;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::core::FixedImpl;

use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16PartialOrd, FP16x16PartialEq, ONE
};

use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};


use orion::operators::tensor::helpers::check_compatibility;

use orion::numbers::fixed_point::implementations::fp16x16::math::{core};

use orion::operators::tensor::{
    core::{ExtraParams, unravel_index},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorAdd, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};

use orion::numbers::fixed_point::{
    implementations::fp16x16::core::{FP16x16Print, FP16x16Div, FP16x16Mul}
};

// Converts a u32 tensor to a FixedPoint tensor.
fn from_u32_to_fixedtype(from_u32: @Tensor<u32>, y_train: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut from_u32_data = *from_u32.data;

    loop {
        match from_u32_data.pop_front() {
            Option::Some(item) => {
                data_result.append(FixedTrait::new_unscaled(*item.into(), false));
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TensorTrait::<FixedType>::new(*y_train.shape, data_result.span(), *y_train.extra)
}


// Calculates the accuracy of the machine learning model's predictions.
fn accuracy(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> FixedType {
    let (mut left, mut right) = (y, z);

    let mut right_data = *right.data;
    let mut left_data = *left.data;
    let mut left_index = 0;
    let mut counter = 0;

    loop {
        match right_data.pop_front() {
            Option::Some(item) => {
                let right_current_index = item;
                let left_current_index = left_data[left_index];

                let (y_value, z_value) = (left_current_index, right_current_index);

                //'left_index'.print();
                //left_index.print();
                //'(*y_value)'.print();
                //(*y_value).print();
                //'(*z_value)'.print();
                //(*z_value).print();

                if *y_value == *z_value {
                    counter += 1;
                };

                left_index += 1;
            },
            Option::None(_) => {
                break;
            }
        };
    };

    //'*y.data).len()'.print();
    //(*y.data).len().print();
    //'counter'.print();
    //counter.print();

    (FixedTrait::new_unscaled(counter, false) / FixedTrait::new_unscaled((*y.data).len(), false))
        * FixedTrait::new_unscaled(100, false)
}

// Returns the truth value of (x < y) element-wise.
fn less(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> Tensor<FixedType> {
    //check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data_result2 = ArrayTrait::<FixedType>::new();
    let (mut smaller, mut bigger, retains_input_order) = if (*y.data).len() < (*z.data).len() {
        (y, z, true)
    } else {
        (z, y, false)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        match bigger_data.pop_front() {
            Option::Some(item) => {
                let bigger_current_index = item;
                let smaller_current_index = smaller_data[smaller_index];

                let (y_value, z_value) = if retains_input_order {
                    (smaller_current_index, bigger_current_index)
                } else {
                    (bigger_current_index, smaller_current_index)
                };

                if *y_value < *z_value {
                    data_result.append(FixedTrait::ONE());
                } else {
                    data_result.append(FixedTrait::ZERO());
                };

                smaller_index = (1 + smaller_index) % smaller_data.len();
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(*bigger.shape, data_result.span(), *y.extra);
}


// Returns an element-wise indication of the sign of a number.
fn sign(z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut z_data = *z.data;

    loop {
        match z_data.pop_front() {
            Option::Some(item) => {
                //'(*item.mag)'.print();
                //(*item.mag).print();
                //'(*item.sign)'.print();
                //(*item.sign).print();                
                let result = if *item.sign {
                    FixedTrait::new(ONE, true)
                } else {
                    FixedTrait::new(ONE, false)
                };
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TensorTrait::<FixedType>::new(*z.shape, data_result.span(), *z.extra)
}

// Returns predictions using the machine learning model.
fn pred(x: @Tensor<FixedType>, w: @Tensor<FixedType>) -> Tensor<FixedType> {
    sign(@(x.matmul(w)))
}
