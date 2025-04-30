use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::ScalarValue;

use crate::stats::{Precision, Stat};
use crate::{Array, Encoding};

/// Computes the min and max of an array, returning the (min, max) values
/// If the array is empty or has only nulls, the result is `None`.
pub trait NaNCountFn<A> {
    fn nan_count(&self, array: A) -> VortexResult<Option<usize>>;
}

impl<E: Encoding> NaNCountFn<&dyn Array> for E
where
    E: for<'a> NaNCountFn<&'a E::Array>,
{
    fn nan_count(&self, array: &dyn Array) -> VortexResult<Option<usize>> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        NaNCountFn::nan_count(self, array_ref)
    }
}

/// Computes the nunmber of NaN values in the array
/// This will update the stats set of this array (as a side effect).
pub fn nan_count(array: &dyn Array) -> VortexResult<Option<usize>> {
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(Some(0));
    }

    let nan_count = array
        .statistics()
        .get_as::<usize>(Stat::NaNCount)
        .and_then(Precision::as_exact);

    if let Some(nan_count) = nan_count {
        return Ok(Some(nan_count));
    }

    // Only float arrays can have NaNs
    let nan_count = if !array.dtype().is_float() {
        Some(0)
    } else if let Some(fn_) = array.vtable().nan_count_fn() {
        fn_.nan_count(array)?
    } else {
        let canonical = array.to_canonical()?;
        if let Some(fn_) = canonical.as_ref().vtable().nan_count_fn() {
            fn_.nan_count(canonical.as_ref())?
        } else {
            vortex_bail!(NotImplemented: "nan_count", array.encoding());
        }
    };

    if let Some(nan_count) = nan_count {
        // Update the stats set with the computed min/max
        array.statistics().set(
            Stat::NaNCount,
            Precision::Exact(ScalarValue::from(nan_count)),
        );
    }

    Ok(nan_count)
}
