use std::ops::BitAnd;
use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_array::BooleanArray;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, ConstantArray};
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, fill_null};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef, ArrayStatistics, Canonical, IntoArray, ToCanonical};

/// Keep only the elements for which the corresponding mask value is true.
///
/// # Examples
///
/// ```
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::arrays::{BoolArray, PrimitiveArray};
/// use vortex_array::compute::{ filter, mask};
/// use vortex_mask::Mask;
/// use vortex_scalar::Scalar;
///
/// let array =
///     PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)]);
/// let mask = Mask::try_from(
///     &BoolArray::from_iter([true, false, false, false, true]),
/// )
/// .unwrap();
///
/// let filtered = filter(&array, &mask).unwrap();
/// assert_eq!(filtered.len(), 2);
/// assert_eq!(filtered.scalar_at(0).unwrap(), Scalar::from(Some(0_i32)));
/// assert_eq!(filtered.scalar_at(1).unwrap(), Scalar::from(Some(2_i32)));
/// ```
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(array: &dyn Array, mask: &Mask) -> VortexResult<ArrayRef> {
    FILTER_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), mask.into()],
            options: &(),
        })?
        .unwrap_array()
}

/// The filter [`ComputeFn`].
pub static FILTER_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("filter".into(), ArcRef::new_ref(&Filter));
    for kernel in inventory::iter::<FilterKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

struct Filter;

impl ComputeFnVTable for Filter {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let FilterArgs { array, mask } = FilterArgs::try_from(args)?;

        let true_count = mask.true_count();

        // Fast-path for empty mask.
        if true_count == 0 {
            return Ok(Canonical::empty(array.dtype()).into_array().into());
        }

        // Fast-path for full mask
        if true_count == mask.len() {
            return Ok(array.to_array().into());
        }

        // Since we handle the AllTrue and AllFalse cases in the entry-point filter function,
        // implementations can use `AllOr::expect_some` to unwrap the mixed values variant.
        let values = match &mask {
            Mask::AllTrue(_) => return Ok(array.to_array().into()),
            Mask::AllFalse(_) => return Ok(Canonical::empty(array.dtype()).into_array().into()),
            Mask::Values(values) => values,
        };

        // Check each kernel for the array
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
        log::debug!("No filter implementation found for {}", array.encoding(),);

        let array_ref = array.to_array().into_arrow_preferred()?;
        let mask_array = BooleanArray::new(values.boolean_buffer().clone(), None);
        let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;

        Ok(ArrayRef::from_arrow(filtered, array.dtype().is_nullable()).into())
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

pub trait FilterKernel: Encoding {
    /// Filter an array by the provided predicate.
    ///
    /// Note that the entry-point filter functions handles `Mask::AllTrue` and `Mask::AllFalse`,
    /// leaving only `Mask::Values` to be handled by this function.
    fn filter(&self, array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef>;
}

/// Adapter to convert a [`FilterKernel`] into a [`Kernel`].
#[derive(Debug)]
pub struct FilterKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + FilterKernel> FilterKernelAdapter<E> {
    pub const fn lift(&'static self) -> FilterKernelRef {
        FilterKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + FilterKernel> Kernel for FilterKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = FilterArgs::try_from(args)?;
        let Some(array) = inputs.array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };
        let filtered = E::filter(&self.0, array, inputs.mask)?;
        Ok(Some(filtered.into()))
    }
}

impl TryFrom<&BoolArray> for Mask {
    type Error = VortexError;

    fn try_from(array: &BoolArray) -> Result<Self, Self::Error> {
        if let Some(constant) = array.as_constant() {
            let bool_constant = constant.as_bool();
            if bool_constant.value().unwrap_or(false) {
                return Ok(Self::new_true(array.len()));
            } else {
                return Ok(Self::new_false(array.len()));
            }
        }

        // Extract a boolean buffer, treating null values to false
        let buffer = match array.validity_mask()? {
            Mask::AllTrue(_) => array.boolean_buffer().clone(),
            Mask::AllFalse(_) => return Ok(Self::new_false(array.len())),
            Mask::Values(validity) => validity.boolean_buffer().bitand(array.boolean_buffer()),
        };

        Ok(Self::from_buffer(buffer))
    }
}

impl TryFrom<&dyn Array> for Mask {
    type Error = VortexError;

    /// Converts from a possible nullable boolean array. Null values are treated as false.
    fn try_from(array: &dyn Array) -> Result<Self, Self::Error> {
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", array.dtype());
        }

        // Convert nulls to false first in case this can be done cheaply by the encoding.
        let array = fill_null(array, &Scalar::bool(false, array.dtype().nullability()))?;

        Self::try_from(&array.to_bool()?)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::filter::filter;

    #[test]
    fn test_filter() {
        let items =
            PrimitiveArray::from_option_iter([Some(0i32), None, Some(1i32), None, Some(2i32)])
                .into_array();
        let mask = Mask::try_from(&BoolArray::from_iter([true, false, true, false, true])).unwrap();

        let filtered = filter(&items, &mask).unwrap();
        assert_eq!(
            filtered.to_primitive().unwrap().as_slice::<i32>(),
            &[0i32, 1i32, 2i32]
        );
    }
}
