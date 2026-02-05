// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): REMOVE THIS FILE!

use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::compute::fill_null;
use crate::vtable::VTable;

/// The filter [`ComputeFn`].
static FILTER_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("filter".into(), ArcRef::new_ref(&Filter));
    for kernel in inventory::iter::<FilterKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    FILTER_FN.kernels().len()
}

/// Keep only the elements for which the corresponding mask value is true.
///
/// # Examples
///
/// ```
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{ filter, mask};
/// use vortex_error::VortexResult;
/// use vortex_mask::Mask;
/// use vortex_scalar::Scalar;
///
/// # fn main() -> VortexResult<()> {
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask = Mask::from_iter([true, false, false, false, true]);
///
/// let filtered = filter(array.as_ref(), &mask)?;
/// assert_eq!(filtered.len(), 2);
/// assert_eq!(filtered.scalar_at(0)?, Scalar::from(Some(0_i32)));
/// assert_eq!(filtered.scalar_at(1)?, Scalar::from(Some(2_i32)));
/// # Ok(())
/// # }
/// ```
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    // TODO(connor): Remove this function completely!!!
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

struct Filter;

impl ComputeFnVTable for Filter {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let FilterArgs { array, mask } = FilterArgs::try_from(args)?;

        debug_assert_eq!(
            array.len(),
            mask.len(),
            "Tried to filter an array via a mask with the wrong length"
        );

        let true_count = mask.true_count();

        if let Some(array) = filter_preconditions(array, mask) {
            return Ok(array.into());
        }

        // If the entire array is null, then we only need to adjust the length of the array.
        if array.validity_mask()?.true_count() == 0 {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), true_count)
                    .into_array()
                    .into(),
            );
        }

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&FILTER_FN, args)? {
            return Ok(output);
        }

        // Otherwise, we can use scalar_at if the mask has length 1.
        if mask.true_count() == 1 {
            let idx = mask.first().vortex_expect("true_count == 1");
            return Ok(ConstantArray::new(array.scalar_at(idx)?, 1)
                .into_array()
                .into());
        }

        // Fallback: implement using Arrow kernels.
        tracing::debug!("No filter implementation found for {}", array.encoding_id(),);

        if !array.is_canonical() {
            let canonical = array.to_canonical()?.into_array();
            return canonical.filter(mask.clone()).map(Into::into);
        };

        vortex_bail!(
            "No filter implementation found for array {}",
            array.encoding_id()
        )
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        Ok(FilterArgs::try_from(args)?.array.dtype().clone())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let FilterArgs { array, mask } = FilterArgs::try_from(args)?;
        if mask.len() != array.len() {
            vortex_bail!(
                "mask.len() is {}, does not equal array.len() of {}",
                mask.len(),
                array.len()
            );
        }
        Ok(mask.true_count())
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

struct FilterArgs<'a> {
    array: &'a dyn Array,
    mask: &'a Mask,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for FilterArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected first input to be an array"))?;
        let mask = value.inputs[1]
            .mask()
            .ok_or_else(|| vortex_err!("Expected second input to be a mask"))?;
        Ok(Self { array, mask })
    }
}

/// A kernel that implements the filter function.
pub struct FilterKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(FilterKernelRef);

pub trait FilterKernel: VTable {
    /// Filter an array by the provided predicate.
    ///
    /// # Preconditions
    ///
    /// The entrypoint filter functions will handle `Mask::AllTrue` and `Mask::AllFalse` on the
    /// selection mask, leaving only `Mask::Values` to be handled by this function.
    ///
    /// Additionally, the array length is guaranteed to have the same length as the selection mask.
    ///
    /// Finally, the array validity mask is guaranteed not to have a true count of 0 (all nulls).
    // TODO(joe): add execution context
    fn filter(&self, array: &Self::Array, selection_mask: &Mask) -> VortexResult<ArrayRef>;
}

/// Adapter to convert a [`FilterKernel`] into a [`Kernel`].
#[derive(Debug)]
pub struct FilterKernelAdapter<V: VTable>(pub V);

impl<V: VTable + FilterKernel> FilterKernelAdapter<V> {
    pub const fn lift(&'static self) -> FilterKernelRef {
        FilterKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + FilterKernel> Kernel for FilterKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = FilterArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        let filtered = <V as FilterKernel>::filter(&self.0, array, inputs.mask)?;
        Ok(Some(filtered.into()))
    }
}

impl dyn Array + '_ {
    /// Converts from a possible nullable boolean array. Null values are treated as false.
    pub fn try_to_mask_fill_null_false(&self) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", self.dtype());
        }

        // Convert nulls to false first in case this can be done cheaply by the encoding.
        let array = fill_null(self, &Scalar::bool(false, self.dtype().nullability()))?;

        Ok(array.to_bool().to_mask_fill_null_false())
    }
}

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
