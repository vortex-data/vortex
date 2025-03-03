use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::arrays::{ConstantArray, NullArray};
use crate::stats::{Precision, Stat, StatsProviderExt};
use crate::{Array, ArrayExt, Encoding};

pub trait IsConstantFn<A> {
    /// # Preconditions
    ///
    /// * All values are valid
    /// * array.len() > 1
    ///
    /// Returns `Ok(None)` to signal we couldn't make an exact determination.
    fn is_constant(&self, array: A, opts: &IsConstantOpts) -> VortexResult<Option<bool>>;
}

impl<E: Encoding> IsConstantFn<&dyn Array> for E
where
    E: for<'a> IsConstantFn<&'a E::Array>,
{
    fn is_constant(&self, array: &dyn Array, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        IsConstantFn::is_constant(self, array_ref, opts)
    }
}

/// Configuration for [`is_constant_opts`] operations.
#[derive(Clone)]
pub struct IsConstantOpts {
    /// Should the operation make an effort to canonicalize the target array if its encoding doesn't implement [`IsConstantFn`].
    pub canonicalize: bool,
}

impl Default for IsConstantOpts {
    fn default() -> Self {
        Self { canonicalize: true }
    }
}

/// Computes whether an array has constant values. If the array's encoding doesn't implement the relevant VTable, it'll try and canonicalize in order to make a determination.
/// An array is constant IFF at least one of the following conditions apply:
/// 1. It has one elements.
/// 1. Its encoded as a [`ConstantArray`] or [`NullArray`]
/// 1. Has an exact statistic attached to it, saying its constant.
/// 1. Is all invalid.
/// 1. Is all valid AND has minimum and maximum statistics that are equal.
///
/// If the array has some null values but is not all null, it'll never be constant.
/// **Please note:** Might return false negatives if a specific encoding couldn't make a determination.
pub fn is_constant(array: &dyn Array) -> VortexResult<bool> {
    let opts = IsConstantOpts::default();
    is_constant_opts(array, &opts)
}

/// Computes whether an array has constant values. Configurable by [`IsConstantOpts`].
///
/// Please see [`is_constant`] for a more detailed explanation of its behavior.
pub fn is_constant_opts(array: &dyn Array, opts: &IsConstantOpts) -> VortexResult<bool> {
    match array.len() {
        // Our current semantics are that we can always get a value out of a constant array. We might want to change that in the future.
        0 => return Ok(false),
        // Array of length 1 is always constant.
        1 => return Ok(true),
        _ => {}
    }

    // Constant and null arrays are always constant
    if array.as_opt::<ConstantArray>().is_some() || array.as_opt::<NullArray>().is_some() {
        return Ok(true);
    }

    // We try and rely on some easy to get stats
    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsConstant) {
        return Ok(value);
    }

    let all_invalid = array.all_invalid()?;
    if all_invalid {
        return Ok(true);
    }

    let all_valid = array.all_valid()?;

    // If we have some nulls, array can't be constant
    if !all_valid && !all_invalid {
        return Ok(false);
    }

    // We already know here that the array is all valid, so we check for min/max stats.
    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(|p| p.as_exact());
    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(|p| p.as_exact());

    if let Some((min, max)) = min.zip(max) {
        if min == max {
            return Ok(true);
        }
    }

    debug_assert!(
        all_valid,
        "All values must be valid as an invariant of the VTable."
    );
    let is_constant = if let Some(vtable_fn) = array.vtable().is_constant_fn() {
        vtable_fn.is_constant(array, opts)?
    } else {
        log::debug!(
            "No is_constant implementation found for {}",
            array.encoding()
        );

        if opts.canonicalize {
            let array = array.to_canonical()?;

            if let Some(is_constant_fn) = array.as_ref().vtable().is_constant_fn() {
                is_constant_fn.is_constant(array.as_ref(), opts)?
            } else {
                vortex_bail!(
                    "No is_constant function for canonical array: {}",
                    array.as_ref().encoding(),
                )
            }
        } else {
            None
        }
    };

    if let Some(is_constant) = is_constant {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(is_constant.into()));
    }

    Ok(is_constant.unwrap_or_default())
}
