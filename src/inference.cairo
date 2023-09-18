use orion::operators::{
    tensor::{core::{Tensor, TensorTrait}, implementations::impl_tensor_fp::Tensor_fp}
};

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use debug::PrintTrait;

use orion::numbers::fixed_point::implementations::fp16x16::core::{ONE, FP16x16Impl, FP16x16Print};

fn sign(mut z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut z_data = *z.data;

    loop {
        match z_data.pop_front() {
            Option::Some(item) => {
                let result = if *item.mag == 0 {
                    FixedTrait::ZERO()
                } else {
                    FixedTrait::ONE()
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

fn pred(x: Tensor<FixedType>, w: Tensor<FixedType>) -> Tensor<FixedType> {
    sign(@x).matmul(@sign(@w))
}
