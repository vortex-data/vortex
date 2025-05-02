use std::any::Any;
use std::sync::LazyLock;

use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arcref::ArcRef;
use crate::arrays::{ConstantArray, NullArray};
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Options, Output};
use crate::stats::{Precision, Stat};
use crate::{Array, ArrayExt, Encoding};

pub fn is_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, false)
}

pub fn is_strict_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, true)
}

pub fn is_sorted_opts(array: &dyn Array, strict: bool) -> VortexResult<bool> {
    Ok(IS_SORTED_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &IsSortedOptions { strict },
        })?
        .unwrap_scalar()?
        .as_bool()
        .value()
        .vortex_expect("non-nullable"))
}

struct IsSorted;
impl ComputeFnVTable for IsSorted {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let IsSortedArgs { array, strict } = IsSortedArgs::try_from(args)?;

        // We currently don't support sorting struct arrays.
        if array.dtype().is_struct() {
            return Ok(Scalar::from(false).into());
        }

        let is_sorted = if strict {
            if let Some(Precision::Exact(value)) =
                array.statistics().get_as::<bool>(Stat::IsStrictSorted)
            {
                return Ok(Scalar::from(value).into());
            }

            let is_strict_sorted = is_sorted_impl(array, kernels, true)?;
            let array_stats = array.statistics();

            if is_strict_sorted {
                array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
                array_stats.set(Stat::IsStrictSorted, Precision::Exact(true.into()));
            } else {
                array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
            }

            is_strict_sorted
        } else {
            if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsSorted)
            {
                return Ok(Scalar::from(value).into());
            }

            let is_sorted = is_sorted_impl(array, kernels, false)?;
            let array_stats = array.statistics();

            if is_sorted {
                array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
            } else {
                array_stats.set(Stat::IsSorted, Precision::Exact(false.into()));
                array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
            }

            is_sorted
        };

        Ok(Scalar::from(is_sorted).into())
    }

    fn return_dtype(&self, _args: &InvocationArgs) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct IsSortedArgs<'a> {
    array: &'a dyn Array,
    strict: bool,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for IsSortedArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 1 {
            vortex_bail!(
                "IsSorted function requires exactly one argument, got {}",
                value.inputs.len()
            );
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Invalid argument type for is sorted function"))?;
        let options = *value
            .options
            .as_any()
            .downcast_ref::<IsSortedOptions>()
            .ok_or_else(|| vortex_err!("Invalid options type for is sorted function"))?;

        Ok(IsSortedArgs {
            array,
            strict: options.strict,
        })
    }
}

#[derive(Clone, Copy)]
struct IsSortedOptions {
    strict: bool,
}

impl Options for IsSortedOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub static IS_SORTED_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("is_sorted".into(), ArcRef::new_ref(&IsSorted));
    for kernel in inventory::iter::<IsSortedKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct IsSortedKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(IsSortedKernelRef);

#[derive(Debug)]
pub struct IsSortedKernelAdapter<E: Encoding>(pub E);

impl<E: Encoding + IsSortedKernel> IsSortedKernelAdapter<E> {
    pub const fn lift(&'static self) -> IsSortedKernelRef {
        IsSortedKernelRef(ArcRef::new_ref(self))
    }
}

impl<E: Encoding + IsSortedKernel> Kernel for IsSortedKernelAdapter<E> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let IsSortedArgs { array, strict } = IsSortedArgs::try_from(args)?;
        let Some(array) = array.as_any().downcast_ref::<E::Array>() else {
            return Ok(None);
        };

        let is_sorted = if strict {
            E::is_strict_sorted(&self.0, array)?
        } else {
            E::is_sorted(&self.0, array)?
        };

        Ok(Some(Scalar::from(is_sorted).into()))
    }
}

pub trait IsSortedKernel: Encoding {
    /// # Preconditions
    /// - The array's length is > 1.
    /// - The array is not encoded as `NullArray` or `ConstantArray`.
    /// - If doing a `strict` check, if the array is nullable, it'll have at most 1 null element
    ///   as the first item in the array.
    fn is_sorted(&self, array: &Self::Array) -> VortexResult<bool>;

    fn is_strict_sorted(&self, array: &Self::Array) -> VortexResult<bool>;
}

#[allow(clippy::wrong_self_convention)]
/// Helper methods to check sortedness with strictness
pub trait IsSortedIteratorExt: Iterator
where
    <Self as Iterator>::Item: PartialOrd,
{
    fn is_strict_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_by(|a, b| a < b)
    }
}

impl<T> IsSortedIteratorExt for T
where
    T: Iterator + ?Sized,
    T::Item: PartialOrd,
{
}

fn is_sorted_impl(
    array: &dyn Array,
    kernels: &[ArcRef<dyn Kernel>],
    strict: bool,
) -> VortexResult<bool> {
    // Arrays with 0 or 1 elements are strict sorted.
    if array.len() <= 1 {
        return Ok(true);
    }

    // Constant and null arrays are always sorted, but not strict sorted.
    if array.is::<ConstantArray>() || array.is::<NullArray>() {
        return Ok(!strict);
    }

    let invalid_count = array.invalid_count()?;

    // Enforce strictness before we even try to check if the array is sorted.
    if strict {
        match invalid_count {
            // We can keep going
            0 => {}
            // If we have a potential null value - it has to be the first one.
            1 => {
                if !array.is_invalid(0)? {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
    }

    let args = InvocationArgs {
        inputs: &[array.into()],
        options: &IsSortedOptions { strict },
    };

    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return Ok(output
                .unwrap_scalar()?
                .as_bool()
                .value()
                .vortex_expect("non-nullable"));
        }
    }
    if let Some(output) = array.invoke(&IS_SORTED_FN, &args)? {
        return Ok(output
            .unwrap_scalar()?
            .as_bool()
            .value()
            .vortex_expect("non-nullable"));
    }

    if !array.is_canonical() {
        log::debug!("No is_sorted implementation found for {}", array.encoding());

        // Recurse to canonical implementation
        let array = array.to_canonical()?;

        return if strict {
            is_strict_sorted(array.as_ref())
        } else {
            is_sorted(array.as_ref())
        };
    }

    vortex_bail!(
        "No is_sorted function for canonical array: {}",
        array.encoding(),
    )
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::Array;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::{is_sorted, is_strict_sorted};
    use crate::validity::Validity;

    #[test]
    fn test_is_sorted() {
        assert!(
            is_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::AllValid
            ))
            .unwrap()
        );
        assert!(
            is_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array())
            ))
            .unwrap()
        );
        assert!(
            !is_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array())
            ))
            .unwrap()
        );

        assert!(
            !is_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 3, 2),
                Validity::AllValid
            ))
            .unwrap()
        );
        assert!(
            !is_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 3, 2),
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            ))
            .unwrap(),
        );
    }

    #[test]
    fn test_is_strict_sorted() {
        assert!(
            is_strict_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::AllValid
            ))
            .unwrap()
        );
        assert!(
            is_strict_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array())
            ))
            .unwrap()
        );
        assert!(
            !is_strict_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 2, 3),
                Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
            ))
            .unwrap(),
        );

        assert!(
            !is_strict_sorted(&PrimitiveArray::new(
                buffer!(0, 1, 3, 2),
                Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
            ))
            .unwrap(),
        );
    }
}
