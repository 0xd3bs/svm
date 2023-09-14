use array::ArrayTrait;
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams}, implementations::impl_tensor_fp::Tensor_fp
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl}, implementations::fp16x16::core::FP16x16Impl
};

fn Y_test() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::new();
    shape.append(50);
    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, true));
    data.append(FixedTrait::new(65536, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(shape.span(), data.span(), Option::Some(extra));
    return tensor;
}
