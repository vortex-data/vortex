use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::stats::{Precision, Stat, StatsProviderExt};
use crate::{Array, Encoding};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult {
    pub min: Scalar,
    pub max: Scalar,
}

/// Computes the min and max of an array, returning the (min, max) values
/// If the array is empty or has only nulls, the result is `None`.
pub trait MinMaxFn<A> {
    fn min_max(&self, array: A) -> VortexResult<Option<MinMaxResult>>;
}

impl<E: Encoding> MinMaxFn<&dyn Array> for E
where
    E: for<'a> MinMaxFn<&'a E::Array>,
{
    fn min_max(&self, array: &dyn Array) -> VortexResult<Option<MinMaxResult>> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        MinMaxFn::min_max(self, array_ref)
    }
}

/// Computes the min & max of an array, returning the (min, max) values
/// The return values are (min, max) scalars, where None indicates that the value is non-existent
/// (e.g. for an empty array).
/// The return value dtype is the non-nullable version of the array dtype.
///
/// This will update the stats set of this array (as a side effect).
pub fn min_max(array: &dyn Array) -> VortexResult<Option<MinMaxResult>> {
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(None);
    }

    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(Precision::as_exact);
    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(Precision::as_exact);

    if let Some((min, max)) = min.zip(max) {
        return Ok(Some(MinMaxResult { min, max }));
    }

    let min_max = if let Some(fn_) = array.vtable().min_max_fn() {
        fn_.min_max(array)?
    } else {
        let canonical = array.to_canonical()?;
        if let Some(fn_) = canonical.as_ref().vtable().min_max_fn() {
            fn_.min_max(canonical.as_ref())?
        } else {
            vortex_bail!(NotImplemented: "min_max", array.encoding());
        }
    };

    if let Some(MinMaxResult { min, max }) = min_max.as_ref() {
        assert_eq!(
            min.dtype(),
            array.dtype(),
            "MinMax min dtype mismatch {}",
            array.encoding()
        );

        assert_eq!(
            max.dtype(),
            array.dtype(),
            "MinMax max dtype mismatch {}",
            array.encoding()
        );

        assert!(
            min <= max,
            "min > max: min={} max={} encoding={}",
            min,
            max,
            array.encoding()
        );

        // Update the stats set with the computed min/max
        array
            .statistics()
            .set(Stat::Min, Precision::Exact(min.value().clone()));
        array
            .statistics()
            .set(Stat::Max, Precision::Exact(max.value().clone()));
    }

    Ok(min_max)
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_buffer::buffer;

    use crate::arrays::{BoolArray, NullArray, PrimitiveArray};
    use crate::compute::{MinMaxResult, min_max};
    use crate::validity::Validity;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable);
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: 1.into(),
                max: 3.into()
            })
        );
    }

    #[test]
    fn test_bool_max() {
        let p = BoolArray::new(
            BooleanBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, false, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, true, false].as_slice()),
            Validity::NonNullable,
        );
        assert_eq!(
            min_max(&p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: true.into()
            })
        );
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(&p).unwrap(), None);
    }
}
