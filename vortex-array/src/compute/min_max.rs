use vortex_error::{vortex_bail, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::stats::{Precision, Stat, Statistics};
use crate::{Array, Encoding, IntoCanonical};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult {
    pub min: Scalar,
    pub max: Scalar,
}

/// Computes the min and max of an array, returning the (min, max) values
/// If the array is empty or has only nulls, the result is `None`.
pub trait MinMaxFn<A> {
    fn min_max(&self, array: &A) -> VortexResult<Option<MinMaxResult>>;
}

impl<E: Encoding> MinMaxFn<Array> for E
where
    E: MinMaxFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn min_max(&self, array: &Array) -> VortexResult<Option<MinMaxResult>> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        MinMaxFn::min_max(encoding, array_ref)
    }
}

/// Computes the min & max of an array, returning the (min, max) values
/// The return values are (min, max) scalars, where None indicates that the value is non-existent
/// (e.g. for an empty array)
/// The return value dtype is the non-nullable version of the array dtype
///
/// This will update the stats set of this array (as a side effect).
pub fn min_max(array: impl AsRef<Array>) -> VortexResult<Option<MinMaxResult>> {
    let array = array.as_ref();

    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(Precision::some_exact);
    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(Precision::some_exact);

    if let Some((min, max)) = min.zip(max) {
        return Ok(Some(MinMaxResult { min, max }));
    }

    let min_max = if let Some(fn_) = array.vtable().min_max_fn() {
        fn_.min_max(array)?
    } else {
        let canonical = array.clone().into_canonical()?;
        if let Some(fn_) = canonical.vtable().min_max_fn() {
            fn_.min_max(canonical.as_ref())?
        } else {
            vortex_bail!(NotImplemented: "min_max", array.encoding());
        }
    };

    if let Some(MinMaxResult { min, max }) = min_max.as_ref() {
        debug_assert_eq!(
            min.dtype(),
            array.dtype(),
            "MinMax min dtype mismatch {}",
            array.encoding()
        );

        array.set_stat(Stat::Min, Precision::exact(min.clone().into_value()));

        debug_assert_eq!(
            max.dtype(),
            array.dtype(),
            "MinMax max dtype mismatch {}",
            array.encoding()
        );
        array.set_stat(Stat::Max, Precision::exact(max.clone().into_value()));

        debug_assert!(
            min <= max,
            "min > max: min={} max={} encoding={}",
            min,
            max,
            array.encoding()
        );
    }

    Ok(min_max)
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;

    use crate::arrays::{BoolArray, NullArray, PrimitiveArray};
    use crate::compute::{min_max, MinMaxResult};
    use crate::validity::Validity::NonNullable;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], NonNullable);
        assert_eq!(
            min_max(p).unwrap(),
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
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, false, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, true, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            Some(MinMaxResult {
                min: false.into(),
                max: true.into()
            })
        );
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(p).unwrap(), None);
    }
}
