use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::builders::ArrayBuilder;
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

    /// Has the same semantics as `Self::take` but materializes the result into the provided
    /// builder.
    fn take_into(
        &self,
        array: A,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(&self.take(array, indices)?)
    }
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

    fn take_into(
        &self,
        array: &dyn Array,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        TakeFn::take_into(self, array_ref, indices, builder)
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

    debug_assert_eq!(
        taken.len(),
        indices.len(),
        "Take length mismatch {}",
        array.encoding()
    );
    #[cfg(debug_assertions)]
    {
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
    }

    Ok(taken)
}

pub fn take_into(
    array: &dyn Array,
    indices: &dyn Array,
    builder: &mut dyn ArrayBuilder,
) -> VortexResult<()> {
    if array.is_empty() && !indices.is_empty() {
        vortex_bail!("Cannot take_into from an empty array");
    }

    #[cfg(debug_assertions)]
    {
        // If either the indices or the array are nullable, the result should be nullable.
        let expected_nullability = indices.dtype().nullability() | array.dtype().nullability();
        assert_eq!(
            builder.dtype(),
            &array.dtype().with_nullability(expected_nullability),
            "Take_into result ({}) should be nullable if, and only if, either the indices ({}) or the array ({}) are nullable. ({})",
            builder.dtype(),
            indices.dtype().nullability().verbose_display(),
            array.dtype().nullability().verbose_display(),
            array.encoding(),
        );
    }

    if !indices.dtype().is_int() {
        vortex_bail!(
            "Take indices must be an integer type, got {}",
            indices.dtype()
        );
    }

    let before_len = builder.len();

    // We know that constant array don't need stats propagation, so we can avoid the overhead of
    // computing derived stats and merging them in.
    take_into_impl(array, indices, builder)?;

    let after_len = builder.len();

    debug_assert_eq!(
        after_len - before_len,
        indices.len(),
        "Take_into length mismatch {}",
        array.encoding()
    );

    Ok(())
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

fn take_into_impl(
    array: &dyn Array,
    indices: &dyn Array,
    builder: &mut dyn ArrayBuilder,
) -> VortexResult<()> {
    let result_nullability = array.dtype().nullability() | indices.dtype().nullability();
    let result_dtype = array.dtype().with_nullability(result_nullability);
    if &result_dtype != builder.dtype() {
        vortex_bail!(
            "TakeIntoFn {} had a builder with a different dtype {} to the resulting array dtype {}",
            array.encoding(),
            builder.dtype(),
            result_dtype,
        );
    }
    if let Some(take_fn) = array.vtable().take_fn() {
        return take_fn.take_into(array, indices, builder);
    }

    // Otherwise, flatten and try again.
    log::debug!("No take_into implementation found for {}", array.encoding());
    let canonical = array.to_canonical()?.into_array();
    let vtable = canonical.vtable();
    let canonical_take_fn = vtable
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding()))?;

    canonical_take_fn.take_into(&canonical, indices, builder)
}
