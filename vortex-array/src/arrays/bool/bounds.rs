// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whether a boolean array is all true or contains a true value, computed directly from its bits.
//!
//! Validity checks need these answers for every array. They are computed here without the
//! aggregate function registry, so they work with any session. Exact cached `Min`/`Max`
//! statistics are used when present, and computed answers are written back to the cache.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;

/// The minimum and maximum of the valid values of a boolean array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolBounds {
    /// Whether every valid value is true.
    pub all_true: bool,
    /// Whether any valid value is true.
    pub any_true: bool,
}

/// Returns the [`BoolBounds`] of the valid values of `array`, or `None` if it has no valid values.
///
/// # Errors
///
/// Fails if `array` is not a boolean array, or if it cannot be executed to a [`BoolArray`].
pub fn bool_bounds(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<BoolBounds>> {
    vortex_ensure!(
        array.dtype().is_boolean(),
        "boolean bounds require a boolean array, got {}",
        array.dtype()
    );

    let stats = array.statistics();
    if let Some(all_true) = stats.get_as::<bool>(Stat::Min).as_exact()
        && let Some(any_true) = stats.get_as::<bool>(Stat::Max).as_exact()
    {
        return Ok(Some(BoolBounds { all_true, any_true }));
    }

    let bools = array.clone().execute::<BoolArray>(ctx)?;
    let values = bools.to_bit_buffer();
    let (valid, true_count) = match BoolArrayExt::validity(&bools).execute_mask(bools.len(), ctx)? {
        Mask::AllTrue(len) => (len, values.true_count()),
        Mask::AllFalse(_) => return Ok(None),
        Mask::Values(mask) => (
            mask.true_count(),
            (&values & mask.bit_buffer()).true_count(),
        ),
    };
    if valid == 0 {
        return Ok(None);
    }

    let bounds = BoolBounds {
        all_true: true_count == valid,
        any_true: true_count > 0,
    };
    stats.set(Stat::Min, Precision::exact(bounds.all_true));
    stats.set(Stat::Max, Precision::exact(bounds.any_true));
    Ok(Some(bounds))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::BoolBounds;
    use super::bool_bounds;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn bounds_of_mixed_values() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            bool_bounds(&array, &mut ctx)?,
            Some(BoolBounds {
                all_true: false,
                any_true: true
            })
        );
        Ok(())
    }

    #[test]
    fn bounds_skip_nulls() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(true)]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            bool_bounds(&array, &mut ctx)?,
            Some(BoolBounds {
                all_true: true,
                any_true: true
            })
        );
        Ok(())
    }

    #[test]
    fn bounds_of_all_null_array() -> VortexResult<()> {
        let array = BoolArray::from_iter([None::<bool>, None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(bool_bounds(&array, &mut ctx)?, None);
        Ok(())
    }
}
