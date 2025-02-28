use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{ConstantArray, NullArray};
use crate::stats::{Precision, Stat};
use crate::{Array, ArrayExt, Encoding};

pub trait IsSortedFn<A> {
    /// # Preconditions
    /// Array is not `NullArray` or `ConstantArray`, and has length > 1.
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

pub fn is_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, false)
}
pub fn is_strict_sorted(array: &dyn Array) -> VortexResult<bool> {
    is_sorted_opts(array, true)
}
pub fn is_sorted_opts(array: &dyn Array, strict: bool) -> VortexResult<bool> {
    // Arrays with 0 or 1 elements are strict sorted.
    if array.len() <= 1 {
        return Ok(true);
    }

    // Constant and null arrays are always sorted, but not strict sorted.
    if array.is::<ConstantArray>() || array.is::<NullArray>() {
        return Ok(!strict);
    }

    // If all values are null, the array is always strictly sorted.
    if array.all_invalid()? {
        return Ok(true);
    }

    let target_stat = if strict {
        Stat::IsStrictSorted
    } else {
        Stat::IsSorted
    };

    // We try and rely on some easy to get stats
    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(target_stat) {
        return Ok(value);
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
