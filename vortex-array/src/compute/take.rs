use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::array::ConstantArray;
use crate::builders::ArrayBuilder;
use crate::encoding::Encoding;
use crate::stats::{Max, Precision, Stat, Statistics, StatsSet};
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

    /// Has the same semantics as `Self::take` but materializes the result into the provided
    /// builder.
    fn take_into(
        &self,
        array: &A,
        indices: &Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(self.take(array, indices)?)
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

    fn take_into(
        &self,
        array: &Array,
        indices: &Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        TakeFn::take_into(encoding, array_ref, indices, builder)
    }
}

pub fn take(array: impl AsRef<Array>, indices: impl AsRef<Array>) -> VortexResult<Array> {
    // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
    //  the filter function since they're typically optimised for this case.
    // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
    //  such that canonicalize does less work.

    let array = array.as_ref();
    let indices = indices.as_ref();

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

    // If the indices are all within bounds, we can skip bounds checking.
    let checked_indices = indices
        .statistics()
        .get_as_bound::<Max, usize>()
        .is_some_and(|max| max < array.len());

    // We know that constant array don't need stats propagation, so we can avoid the overhead of
    // computing derived stats and merging them in.
    let derived_stats = (!array.must_be_constant()).then(|| derive_take_stats(array));

    let taken = take_impl(array, indices, checked_indices)?;

    if let Some(derived_stats) = derived_stats {
        let mut stats = taken.stats_set();
        stats.combine_sets(&derived_stats, array.dtype())?;
        for (stat, val) in stats.into_iter() {
            taken.set_stat(stat, val)
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
        let expected_nullability =
            (indices.dtype().is_nullable() || array.dtype().is_nullable()).into();
        assert_eq!(
            taken.dtype(),
            &array.dtype().with_nullability(expected_nullability),
            "Take result should be nullable if either the indices or the array are nullable"
        );
    }

    Ok(taken)
}

pub fn take_into(
    array: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    builder: &mut dyn ArrayBuilder,
) -> VortexResult<()> {
    let array = array.as_ref();
    let indices = indices.as_ref();

    #[cfg(debug_assertions)]
    {
        // If either the indices or the array are nullable, the result should be nullable.
        let expected_nullability =
            (indices.dtype().is_nullable() || array.dtype().is_nullable()).into();
        assert_eq!(
            builder.dtype(),
            &array.dtype().with_nullability(expected_nullability),
            "Take_into result should be nullable if either the indices or the array are nullable"
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

fn derive_take_stats(arr: &Array) -> StatsSet {
    let stats = arr.stats_set();

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

fn take_into_impl(
    array: &Array,
    indices: &Array,
    builder: &mut dyn ArrayBuilder,
) -> VortexResult<()> {
    if array.dtype() != builder.dtype() {
        vortex_bail!(
            "TakeIntoFn {} had a builder with a different dtype {} to the array dtype {}",
            array.encoding(),
            array.dtype(),
            builder.dtype()
        );
    }
    if let Some(take_fn) = array.vtable().take_fn() {
        return take_fn.take_into(array, indices, builder);
    }

    // Otherwise, flatten and try again.
    log::debug!("No take_into implementation found for {}", array.encoding());
    let canonical = array.clone().into_canonical()?.into_array();
    let canonical_take_fn = canonical
        .vtable()
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding()))?;

    canonical_take_fn.take_into(&canonical, indices, builder)
}
