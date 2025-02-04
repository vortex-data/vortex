use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{Max, Stat, Statistics, StatsSet};
use crate::{Array, IntoArray, IntoCanonical};

pub trait TakeFn<A> {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Panics
    ///
    /// Using `indices` that are invalid for the given `array` will cause a panic.
    fn take(&self, array: &A, indices: &Array) -> VortexResult<Array>;

    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Safety
    ///
    /// This take variant will not perform bounds checking on indices, so it is the caller's
    /// responsibility to ensure that the `indices` are all valid for the provided `array`.
    /// Failure to do so could result in out of bounds memory access or UB.
    unsafe fn take_unchecked(&self, array: &A, indices: &Array) -> VortexResult<Array> {
        self.take(array, indices)
    }
}

impl<E: Encoding> TakeFn<Array> for E
where
    E: TakeFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn take(&self, array: &Array, indices: &Array) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        TakeFn::take(encoding, array_ref, indices)
    }
}

pub fn take(array: impl AsRef<Array>, indices: impl AsRef<Array>) -> VortexResult<Array> {
    // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
    //  the filter function since they're typically optimised for this case.
    // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
    //  such that canonicalize does less work.

    let array = array.as_ref();
    let indices = indices.as_ref();

    if !indices.dtype().is_int() || indices.dtype().is_nullable() {
        vortex_bail!(
            "Take indices must be a non-nullable integer type, got {}",
            indices.dtype()
        );
    }

    // If the indices are all within bounds, we can skip bounds checking.
    let checked_indices = indices
        .statistics()
        .get_as_bound::<Max, usize>()
        .is_some_and(|max| max < array.len());

    let derived_stats = derive_take_stats(array);

    let taken = take_impl(array, indices, checked_indices)?;

    let mut stats = taken.stats_set();
    stats.combine_sets(&derived_stats, array.dtype())?;
    // TODO(joe): add
    // taken.inherit_statistics(&stats)?;
    for (stat, val) in stats.iter() {
        taken.statistics().set_stat(*stat, val.clone())
    }

    debug_assert_eq!(
        taken.len(),
        indices.len(),
        "Take length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        array.dtype(),
        taken.dtype(),
        "Take dtype mismatch {}",
        array.encoding()
    );

    Ok(taken)
}

fn derive_take_stats(arr: &Array) -> StatsSet {
    let stats = arr.stats_set();

    stats.keep_exact_inexact_stats(
        // Any combination of elements from a constant array is still const
        &[Stat::IsConstant],
        &[
            // Cannot create values smaller than min or larger than max
            Stat::Min,
            Stat::Max,
        ],
    )
}

fn take_impl(array: &Array, indices: &Array, checked_indices: bool) -> VortexResult<Array> {
    // If TakeFn defined for the encoding, delegate to TakeFn.
    // If we know from stats that indices are all valid, we can avoid all bounds checks.
    if let Some(take_fn) = array.vtable().take_fn() {
        let result = if checked_indices {
            // SAFETY: indices are all inbounds per stats.
            // TODO(aduffy): this means stats must be trusted, can still trigger UB if stats are bad.
            unsafe { take_fn.take_unchecked(array, indices) }
        } else {
            take_fn.take(array, indices)
        }?;
        if array.dtype() != result.dtype() {
            vortex_bail!(
                "TakeFn {} changed array dtype from {} to {}",
                array.encoding(),
                array.dtype(),
                result.dtype()
            );
        }
        return Ok(result);
    }

    // Otherwise, flatten and try again.
    log::debug!("No take implementation found for {}", array.encoding());
    let canonical = array.clone().into_canonical()?.into_array();
    let canonical_take_fn = canonical
        .vtable()
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding()))?;

    if checked_indices {
        // SAFETY: indices are known to be in-bound from stats
        unsafe { canonical_take_fn.take_unchecked(&canonical, indices) }
    } else {
        canonical_take_fn.take(&canonical, indices)
    }
}
