// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arcref::ArcRef;
use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::vtable::VTable;

/// Keep only the elements for which the corresponding mask value is true.
///
/// This is a convenience function that wraps `array.filter(mask)` and canonicalizes the result.
pub fn filter(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    Ok(array.filter(mask.clone())?.to_canonical()?.into_array())
}

///  The set of common preconditions that apply to all arrays.
pub fn filter_preconditions(array: &dyn Array, mask: &Mask) -> Option<ArrayRef> {
    let true_count = mask.true_count();
    // Fast-path for empty mask.
    if true_count == 0 {
        return Some(Canonical::empty(array.dtype()).into_array());
    }

    // Fast-path for full mask
    if true_count == mask.len() {
        return Some(array.to_array());
    }

    None
}

/// Filter an array using Arrow's filter kernel.
pub fn arrow_filter_fn(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    let values = match &mask {
        Mask::Values(values) => values,
        Mask::AllTrue(_) | Mask::AllFalse(_) => unreachable!("check in filter invoke"),
    };

    let array_ref = array.to_array().into_arrow_preferred()?;
    let mask_array = BooleanArray::new(values.bit_buffer().clone().into(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

    ArrayRef::from_arrow(filtered.as_ref(), array.dtype().is_nullable())
}

impl dyn Array + '_ {
    /// Converts from a possible nullable boolean array. Null values are treated as false.
    pub fn try_to_mask_fill_null_false(&self) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", self.dtype());
        }

        // Convert nulls to false first in case this can be done cheaply by the encoding.
        let array =
            crate::compute::fill_null(self, &Scalar::bool(false, self.dtype().nullability()))?;

        Ok(array.to_bool().to_mask_fill_null_false())
    }
}

// =============================================================================
// Legacy FilterKernel infrastructure (to be removed after migration)
// =============================================================================

/// A kernel that implements the filter function.
///
/// NOTE: This is legacy infrastructure. The kernel registration is disabled.
/// These implementations will be removed after verifying the new execute infrastructure
/// handles each array type correctly.
pub struct FilterKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(FilterKernelRef);

/// Legacy trait for implementing filter on specific array types.
///
/// NOTE: This is being replaced by the FilterArray execute infrastructure.
pub trait FilterKernel: VTable {
    /// Filter an array by the provided predicate.
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef>;
}

/// Adapter to lift a FilterKernel implementation into a Kernel.
#[derive(Debug)]
pub struct FilterKernelAdapter<V: VTable>(pub V);

impl<V: VTable + FilterKernel> FilterKernelAdapter<V> {
    pub const fn lift(&'static self) -> FilterKernelRef {
        FilterKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + FilterKernel> Kernel for FilterKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        // NOTE: This kernel is not actually registered, so this code is never called.
        // It exists only to satisfy the trait bound for FilterKernelRef.
        let _ = args;
        Ok(None)
    }
}
