use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{ConstantArray, NullArray};
use crate::stats::{Precision, Stat};
use crate::{Array, ArrayExt, Encoding};

pub trait IsSortedFn<A> {
    /// # Preconditions
    /// - The array's length is > 1.
    /// - The array is not encoded as `NullArray` or `ConstantArray`.
    /// - If doing a `strict` check, if the array is nullable, it'll have at most 1 null element
    ///   as the first item in the array.
    fn is_sorted(&self, array: A) -> VortexResult<bool>;

    fn is_strict_sorted(&self, array: A) -> VortexResult<bool>;
}

impl<E: Encoding> IsSortedFn<&dyn Array> for E
where
    E: for<'a> IsSortedFn<&'a E::Array>,
{
    fn is_sorted(&self, array: &dyn Array) -> VortexResult<bool> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        IsSortedFn::is_sorted(self, array_ref)
    }

    fn is_strict_sorted(&self, array: &dyn Array) -> VortexResult<bool> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        IsSortedFn::is_strict_sorted(self, array_ref)
    }
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

pub fn is_sorted(array: &dyn Array) -> VortexResult<bool> {
    // We currently don't support sorting struct arrays.
    if array.dtype().is_struct() {
        return Ok(false);
    }

    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsSorted) {
        return Ok(value);
    }

    let is_sorted = is_sorted_impl(array, false)?;
    let array_stats = array.statistics();

    if is_sorted {
        array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
    } else {
        array_stats.set(Stat::IsSorted, Precision::Exact(false.into()));
        array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
    }

    Ok(is_sorted)
}

pub fn is_strict_sorted(array: &dyn Array) -> VortexResult<bool> {
    // We currently don't support sorting struct arrays.
    if array.dtype().is_struct() {
        return Ok(false);
    }

    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsStrictSorted) {
        return Ok(value);
    }

    let is_sorted = is_sorted_impl(array, true)?;
    let array_stats = array.statistics();

    if is_sorted {
        array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
        array_stats.set(Stat::IsStrictSorted, Precision::Exact(true.into()));
    } else {
        array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
    }

    Ok(is_sorted)
}

fn is_sorted_impl(array: &dyn Array, strict: bool) -> VortexResult<bool> {
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

    let is_sorted = if let Some(vtable_fn) = array.vtable().is_sorted_fn() {
        if strict {
            vtable_fn.is_strict_sorted(array)?
        } else {
            vtable_fn.is_sorted(array)?
        }
    } else {
        log::debug!("No is_sorted implementation found for {}", array.encoding());
        let array = array.to_canonical()?;

        if let Some(vtable_fn) = array.as_ref().vtable().is_sorted_fn() {
            let array = array.as_ref();
            if strict {
                vtable_fn.is_strict_sorted(array)?
            } else {
                vtable_fn.is_sorted(array)?
            }
        } else {
            vortex_bail!(
                "No is_sorted function for canonical array: {}",
                array.as_ref().encoding(),
            )
        }
    };

    Ok(is_sorted)
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
            is_sorted(&PrimitiveArray::new(
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
