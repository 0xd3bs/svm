use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16PartialOrd;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::{Tensor, TensorTrait};

use orion::operators::tensor::helpers::check_compatibility;


use orion::operators::tensor::{
    core::{ExtraParams, unravel_index},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorAdd, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};


/// Cf: TensorTrait::less docstring
fn less(y: @Tensor<FixedType>, z: @Tensor<FixedType>) -> Tensor<FixedType> {
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
            data_result.append(FP16x16Impl::ONE());
        } else {
            data_result.append(FP16x16Impl::ZERO());
        };

        smaller_index = (1 + smaller_index) % smaller_data.len();
    };

    return TensorTrait::<FixedType>::new(*bigger.shape, data_result.span(), *y.extra);
}
