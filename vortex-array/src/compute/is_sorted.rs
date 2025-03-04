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
    fn is_sorted(&self, array: A, strict: bool) -> VortexResult<bool>;
}

impl<E: Encoding> IsSortedFn<&dyn Array> for E
where
    E: for<'a> IsSortedFn<&'a E::Array>,
{
    fn is_sorted(&self, array: &dyn Array, strict: bool) -> VortexResult<bool> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        IsSortedFn::is_sorted(self, array_ref, strict)
    }
}

#[allow(clippy::wrong_self_convention)]
/// Helper methods to check sortedness with strictness
pub trait IteratorExt: Iterator
where
    <Self as Iterator>::Item: PartialOrd,
{
    fn is_sorted_with_strictness(self, strict: bool) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        if strict {
            Iterator::is_sorted_by(self, |a, b| a < b)
        } else {
            Iterator::is_sorted(self)
        }
    }

    fn is_strict_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_with_strictness(true)
    }
}

impl<T> IteratorExt for T
where
    T: Iterator + ?Sized,
    T::Item: PartialOrd,
{
}

pub fn is_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, false)
}
pub fn is_strict_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, true)
}

pub fn is_sorted_opts(array: &dyn Array, strict: bool) -> VortexResult<bool> {
    let target_stat = if strict {
        Stat::IsStrictSorted
    } else {
        Stat::IsSorted
    };

    // We try and rely on some easy to get stats
    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(target_stat) {
        return Ok(value);
    }

    let is_sorted = is_sorted_impl(array, strict)?;

    let array_stats = array.statistics();

    match (strict, is_sorted) {
        (true, true) => {
            array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(true.into()));
        }
        (true, false) => {
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
        }
        (false, true) => {
            array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
        }
        (false, false) => {
            array_stats.set(Stat::IsSorted, Precision::Exact(false.into()));
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
        }
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
            1 => return array.is_invalid(0),
            _ => return Ok(false),
        }
    }

    let is_sorted = if let Some(vtable_fn) = array.vtable().is_sorted_fn() {
        vtable_fn.is_sorted(array, strict)?
    } else {
        log::debug!("No is_sorted implementation found for {}", array.encoding());
        let array = array.to_canonical()?;

        if let Some(vtable_fn) = array.as_ref().vtable().is_sorted_fn() {
            vtable_fn.is_sorted(array.as_ref(), strict)?
        } else {
            vortex_bail!(
                "No is_constant function for canonical array: {}",
                array.as_ref().encoding(),
            )
        }
    };

    Ok(is_sorted)
}
