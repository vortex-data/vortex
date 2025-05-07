use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::encoding::Encoding;
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSet};
use crate::{Array, ArrayRef, IntoArray};

pub trait TakeFn<A> {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Panics
    ///
    /// Using `indices` that are invalid for the given `array` will cause a panic.
    fn take(&self, array: A, indices: &dyn Array) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> TakeFn<&dyn Array> for E
where
    E: for<'a> TakeFn<&'a E::Array>,
{
    fn take(&self, array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        TakeFn::take(self, array_ref, indices)
    }
}

pub fn take(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
    //  the filter function since they're typically optimised for this case.
    // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
    //  such that canonicalize does less work.
    if indices.all_invalid()? {
        return Ok(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                .into_array(),
        );
    }

    if !indices.dtype().is_int() {
        vortex_bail!(
            "Take indices must be an integer type, got {}",
            indices.dtype()
        );
    }

    // We know that constant array don't need stats propagation, so we can avoid the overhead of
    // computing derived stats and merging them in.
    let derived_stats = (!array.is_constant()).then(|| derive_take_stats(array));

    let taken = take_impl(array, indices)?;

    if let Some(derived_stats) = derived_stats {
        let mut stats = taken.statistics().to_owned();
        stats.combine_sets(&derived_stats, array.dtype())?;
        for (stat, val) in stats.into_iter() {
            taken.statistics().set(stat, val)
        }
    }

    assert_eq!(
        taken.len(),
        indices.len(),
        "Take length mismatch {}",
        array.encoding()
    );
    // If either the indices or the array are nullable, the result should be nullable.
    let expected_nullability = indices.dtype().nullability() | array.dtype().nullability();
    assert_eq!(
        taken.dtype(),
        &array.dtype().with_nullability(expected_nullability),
        "Take result ({}) should be nullable if either the indices ({}) or the array ({}) are nullable. ({})",
        taken.dtype(),
        indices.dtype().nullability().verbose_display(),
        array.dtype().nullability().verbose_display(),
        array.encoding(),
    );

    Ok(taken)
}

fn derive_take_stats(arr: &dyn Array) -> StatsSet {
    let stats = arr.statistics().to_owned();

    let is_constant = stats.get_as::<bool>(Stat::IsConstant);

    let mut stats = stats.keep_inexact_stats(&[
        // Cannot create values smaller than min or larger than max
        Stat::Min,
        Stat::Max,
    ]);

    if is_constant == Some(Precision::Exact(true)) {
        // Any combination of elements from a constant array is still const
        stats.set(Stat::IsConstant, Precision::exact(true));
    }

    stats
}

fn take_impl(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    // First look for a TakeFrom specialized on the indices.
    if let Some(take_from_fn) = indices.vtable().take_from_fn() {
        if let Some(arr) = take_from_fn.take_from(indices, array)? {
            return Ok(arr);
        }
    }

    // If TakeFn defined for the encoding, delegate to TakeFn.
    // If we know from stats that indices are all valid, we can avoid all bounds checks.
    if let Some(take_fn) = array.vtable().take_fn() {
        return take_fn.take(array, indices);
    }

    // Otherwise, flatten and try again.
    log::debug!("No take implementation found for {}", array.encoding());
    let canonical = array.to_canonical()?.into_array();
    let vtable = canonical.vtable();
    let canonical_take_fn = vtable
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding()))?;

    canonical_take_fn.take(&canonical, indices)
}
