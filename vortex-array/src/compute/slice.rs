use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{Stat, Statistics, StatsSet};
use crate::{Array, Canonical, IntoArray};

/// Limit array to start...stop range
pub trait SliceFn<A> {
    /// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
    /// If start >= stop, returns an empty array of the same type as `self`.
    /// Assumes that start or stop are out of bounds, may panic otherwise.
    fn slice(&self, array: &A, start: usize, stop: usize) -> VortexResult<Array>;
}

impl<E: Encoding> SliceFn<Array> for E
where
    E: SliceFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn slice(&self, array: &Array, start: usize, stop: usize) -> VortexResult<Array> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        SliceFn::slice(encoding, array_ref, start, stop)
    }
}

/// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
///
/// # Errors
///
/// Slicing returns an error if you attempt to slice a range that exceeds the bounds of the
/// underlying array.
///
/// Slicing returns an error if the underlying codec's [slice](SliceFn::slice()) implementation
/// returns an error.
pub fn slice(array: impl AsRef<Array>, start: usize, stop: usize) -> VortexResult<Array> {
    let array = array.as_ref();

    if start == 0 && stop == array.len() {
        return Ok(array.clone());
    }

    if start == stop {
        return Ok(Canonical::empty(array.dtype()).into_array());
    }

    check_slice_bounds(array, start, stop)?;

    let derived_stats = derive_sliced_stats(array);

    let sliced = array
        .vtable()
        .slice_fn()
        .map(|f| f.slice(array, start, stop))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "slice",
                array.encoding()
            ))
        })?;

    let mut stats = sliced.stats_set();
    stats.combine_sets(&derived_stats, array.dtype())?;
    for (stat, val) in stats.into_iter() {
        sliced.set_stat(stat, val)
    }

    debug_assert_eq!(
        sliced.len(),
        stop - start,
        "Slice length mismatch {}",
        array.encoding()
    );
    debug_assert_eq!(
        sliced.dtype(),
        array.dtype(),
        "Slice dtype mismatch {}",
        array.encoding()
    );

    Ok(sliced)
}

fn derive_sliced_stats(arr: &Array) -> StatsSet {
    let stats = arr.stats_set();

    stats.keep_exact_inexact_stats(
        &[Stat::IsConstant, Stat::IsSorted, Stat::IsStrictSorted],
        &[
            Stat::Max,
            Stat::Min,
            Stat::RunCount,
            Stat::TrueCount,
            Stat::NullCount,
            Stat::UncompressedSizeInBytes,
        ],
    )
}

fn check_slice_bounds(array: &Array, start: usize, stop: usize) -> VortexResult<()> {
    if start > array.len() {
        vortex_bail!(OutOfBounds: start, 0, array.len());
    }
    if stop > array.len() {
        vortex_bail!(OutOfBounds: stop, 0, array.len());
    }
    if start > stop {
        vortex_bail!("start ({start}) must be <= stop ({stop})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_scalar::Scalar;

    use crate::array::{ConstantArray, PrimitiveArray};
    use crate::compute::slice;
    use crate::stats::{Precision, Stat, Statistics, STATS_TO_WRITE};

    #[test]
    fn test_slice_primitive() {
        let c = PrimitiveArray::from_iter(0i32..100);
        c.compute_all(STATS_TO_WRITE).unwrap();

        let c2 = slice(c, 10, 20).unwrap();

        let result_stats = c2.stats_set();
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Max),
            Some(Precision::inexact(99))
        );
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Min),
            Some(Precision::inexact(0))
        );
    }

    #[test]
    fn test_slice_const() {
        let c = ConstantArray::new(Scalar::from(10), 100);
        c.compute_all(STATS_TO_WRITE).unwrap();

        let c2 = slice(c, 10, 20).unwrap();
        let result_stats = c2.stats_set();

        // Constant always knows its exact stats
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Max),
            Some(Precision::exact(10))
        );
        assert_eq!(
            result_stats.get_as::<i32>(Stat::Min),
            Some(Precision::exact(10))
        );
    }
}
