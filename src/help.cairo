use debug::PrintTrait;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16PartialOrd;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16PartialEq;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
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
    implementations::fp16x16::core::{
        FP16x16Print
    }
};

fn from_u32_to_fixedtype(from_u32: @Tensor<u32>, y_train: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut from_u32_data = *from_u32.data;

    loop {
        match from_u32_data.pop_front() {
            Option::Some(item) => {
                //let result = if *item.into() == 1_u32 {
                //                FixedTrait::ONE()
                //            } else {
                //                FixedTrait::ZERO()
                //            };
                //data_result.append(result);
                data_result.append(FixedTrait::new_unscaled(*item.into(), false));
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TensorTrait::<FixedType>::new(*y_train.shape, data_result.span(), *y_train.extra)
}

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

fn _less(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> Tensor<FixedType> {
    check_compatibility(*y.shape, *z.shape);

    let mut data_result = ArrayTrait::<FixedType>::new();
    let (mut smaller, mut bigger, retains_input_order) = if (*y.data).len() < (*z.data).len() {
        (y, z, true)
    } else {
        (z, y, false)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        if bigger_data.len() == 0 {
            break ();
        };

        let bigger_current_index = *bigger_data.pop_front().unwrap();
        let smaller_current_index = *smaller_data[smaller_index];

        let (y_value, z_value) = if retains_input_order {
            (smaller_current_index, bigger_current_index)
        } else {
            (bigger_current_index, smaller_current_index)
        };

        if y_value < z_value {
            data_result.append(FixedTrait::ONE());
        } else {
            data_result.append(FixedTrait::ZERO());
        };

        smaller_index = (1 + smaller_index) % smaller_data.len();
    };

    return TensorTrait::<FixedType>::new(*bigger.shape, data_result.span(), *y.extra);
}
