// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer bounds and sortedness for array validation and integer narrowing.
//!
//! These read the array's `Min`, `Max`, `IsSorted` and `IsStrictSorted` statistics, which are
//! cached and computed by the aggregate functions the session registers, using any kernel the
//! encoding provides. When a statistic has no answer, because the session registers no aggregate
//! for it, the array is executed to a [`PrimitiveArray`] and scanned once instead, and the scanned
//! `Min`, `Max` and `NullCount` are cached. Validation therefore works in every session.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// The minimum and maximum of the valid values of an integer array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerBounds {
    /// The smallest valid value.
    pub min: i128,
    /// The largest valid value.
    pub max: i128,
}

/// Returns the [`IntegerBounds`] of the valid values of `array`, or `None` if it has no valid
/// values.
///
/// # Errors
///
/// Fails if `array` is not an integer array, or if computing its statistics or executing it to a
/// [`PrimitiveArray`] fails.
pub fn integer_bounds(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<IntegerBounds>> {
    ensure_int(array)?;
    let stats = array.statistics();
    if let Some(min) = stats
        .compute_stat(Stat::Min, ctx)?
        .and_then(|s| as_i128(array, &s))
        && let Some(max) = stats
            .compute_stat(Stat::Max, ctx)?
            .and_then(|s| as_i128(array, &s))
    {
        return Ok(Some(IntegerBounds { min, max }));
    }
    Ok(scan_array(array, ctx)?.map(|scanned| IntegerBounds {
        min: scanned.min,
        max: scanned.max,
    }))
}

/// Returns whether the valid values of `array` are in non-decreasing order.
///
/// # Errors
///
/// Fails if `array` is not an integer array, or if computing its statistics or executing it to a
/// [`PrimitiveArray`] fails.
pub fn integer_is_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    sortedness(array, Stat::IsSorted, ctx)
}

/// Returns whether the valid values of `array` are in strictly increasing order.
///
/// # Errors
///
/// Fails if `array` is not an integer array, or if computing its statistics or executing it to a
/// [`PrimitiveArray`] fails.
pub fn integer_is_strict_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    sortedness(array, Stat::IsStrictSorted, ctx)
}

fn sortedness(array: &ArrayRef, stat: Stat, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    ensure_int(array)?;
    if let Some(sorted) = array
        .statistics()
        .compute_stat(stat, ctx)?
        .and_then(|s| s.as_bool().value())
    {
        return Ok(sorted);
    }
    // An array without valid values is trivially sorted.
    Ok(scan_array(array, ctx)?.is_none_or(|scanned| match stat {
        Stat::IsStrictSorted => scanned.is_strict_sorted,
        _ => scanned.is_sorted,
    }))
}

fn ensure_int(array: &ArrayRef) -> VortexResult<()> {
    vortex_ensure!(
        array.dtype().is_int(),
        "integer bounds require an integer array, got {}",
        array.dtype()
    );
    Ok(())
}

fn as_i128(array: &ArrayRef, scalar: &Scalar) -> Option<i128> {
    match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
        scalar.as_primitive().as_::<P>().map(i128::from)
    })
}

/// The bounds and sortedness of the valid values of an integer array, widened to `i128`.
struct ScannedBounds {
    min: i128,
    max: i128,
    is_sorted: bool,
    is_strict_sorted: bool,
}

/// Executes `array` to a [`PrimitiveArray`] and scans its valid values once, caching `Min`, `Max`
/// and `NullCount`. Returns `None` if `array` has no valid values.
fn scan_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<ScannedBounds>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    let mask = PrimitiveArrayExt::validity(&primitive).execute_mask(primitive.len(), ctx)?;

    let stats = array.statistics();
    stats.set(
        Stat::NullCount,
        Precision::exact(primitive.len() - mask.true_count()),
    );

    match_each_integer_ptype!(array.dtype().as_ptype(), |P| {
        let Some(scanned) = scan(primitive.as_slice::<P>(), &mask) else {
            return Ok(None);
        };
        stats.set(Stat::Min, Precision::exact(scanned.min));
        stats.set(Stat::Max, Precision::exact(scanned.max));
        Ok(Some(ScannedBounds {
            min: i128::from(scanned.min),
            max: i128::from(scanned.max),
            is_sorted: scanned.is_sorted,
            is_strict_sorted: scanned.is_strict_sorted,
        }))
    })
}

/// The bounds of the values selected by a mask, in the values' own type.
struct Scanned<P> {
    min: P,
    max: P,
    is_sorted: bool,
    is_strict_sorted: bool,
}

/// Returns the minimum, maximum and sortedness of the values selected by `mask`.
fn scan<P: Copy + PartialOrd>(values: &[P], mask: &Mask) -> Option<Scanned<P>> {
    let mut valid = values
        .iter()
        .enumerate()
        .filter(|(idx, _)| match mask {
            Mask::AllTrue(_) => true,
            Mask::AllFalse(_) => false,
            Mask::Values(values) => values.value(*idx),
        })
        .map(|(_, value)| *value);

    let first = valid.next()?;
    let mut scanned = Scanned {
        min: first,
        max: first,
        is_sorted: true,
        is_strict_sorted: true,
    };
    let mut prev = first;
    for value in valid {
        if value < scanned.min {
            scanned.min = value;
        }
        if value > scanned.max {
            scanned.max = value;
        }
        scanned.is_sorted &= prev <= value;
        scanned.is_strict_sorted &= prev < value;
        prev = value;
    }
    Some(scanned)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::IntegerBounds;
    use super::integer_bounds;
    use super::integer_is_sorted;
    use super::integer_is_strict_sorted;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;

    /// A session registering no aggregate functions, so statistics fall back to scanning.
    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn scans_sorted_values() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([0u32, 2, 2, 7]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            integer_bounds(&array, &mut ctx)?,
            Some(IntegerBounds { min: 0, max: 7 })
        );
        assert!(integer_is_sorted(&array, &mut ctx)?);
        assert!(!integer_is_strict_sorted(&array, &mut ctx)?);
        assert_eq!(
            array.statistics().get_as::<u32>(Stat::Max),
            Precision::exact(7u32)
        );
        assert_eq!(
            array.statistics().get_as::<bool>(Stat::IsSorted),
            Precision::Absent
        );
        Ok(())
    }

    #[test]
    fn scan_skips_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(5i64), None, Some(-3), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            integer_bounds(&array, &mut ctx)?,
            Some(IntegerBounds { min: -3, max: 5 })
        );
        assert!(!integer_is_sorted(&array, &mut ctx)?);
        assert_eq!(
            array.statistics().get_as::<usize>(Stat::NullCount),
            Precision::exact(2usize)
        );
        Ok(())
    }

    #[test]
    fn all_null_array_has_no_bounds_and_is_sorted() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([None::<u8>, None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(integer_bounds(&array, &mut ctx)?, None);
        assert!(integer_is_strict_sorted(&array, &mut ctx)?);
        Ok(())
    }

    /// Exact statistics are read instead of scanning. They deliberately disagree with the values,
    /// which a scan would report.
    #[test]
    fn reads_exact_statistics() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([3i32, 1, 2]).into_array();
        let stats = array.statistics();
        stats.set(Stat::Min, Precision::exact(-10i32));
        stats.set(Stat::Max, Precision::exact(10i32));
        stats.set(Stat::IsSorted, Precision::exact(true));
        stats.set(Stat::IsStrictSorted, Precision::exact(true));
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            integer_bounds(&array, &mut ctx)?,
            Some(IntegerBounds { min: -10, max: 10 })
        );
        assert!(integer_is_sorted(&array, &mut ctx)?);
        assert!(integer_is_strict_sorted(&array, &mut ctx)?);
        Ok(())
    }
}
