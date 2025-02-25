use crate::Array;
use crate::arrays::BoolEncoding;
use crate::compute::{
    BinaryBooleanFn, CastFn, FillForwardFn, FillNullFn, FilterFn, InvertFn, IsConstantFn, MaskFn,
    MinMaxFn, ScalarAtFn, SliceFn, SumFn, TakeFn, ToArrowFn, UncompressedSizeFn,
};
use crate::vtable::ComputeVTable;

mod cast;
mod fill_forward;
mod fill_null;
pub mod filter;
mod flatten;
mod invert;
mod is_constant;
mod mask;
mod min_max;
mod scalar_at;
mod slice;
mod sum;
mod take;
mod to_arrow;
mod uncompressed_size;

impl ComputeVTable for BoolEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<&dyn Array>> {
        // We only implement this when other is a constant value, otherwise we fall back to the
        // default implementation that canonicalizes to Arrow.
        // TODO(ngates): implement this for constants.
        // other.is_constant().then_some(self)
        None
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<&dyn Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<&dyn Array>> {
        Some(self)
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
