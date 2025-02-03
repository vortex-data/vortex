use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::stats::Precision;
use crate::{Array, Encoding, IntoCanonical};

pub type MinMaxResult = Option<(Precision<Scalar>, Precision<Scalar>)>;

/// Computes the min and max of an array, returning the (min, max) values
pub trait MinMaxFn<A> {
    fn min_max(&self, array: &A) -> VortexResult<MinMaxResult>;
}

impl<E: Encoding> MinMaxFn<Array> for E
where
    E: MinMaxFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn min_max(&self, array: &Array) -> VortexResult<MinMaxResult> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        MinMaxFn::min_max(encoding, array_ref)
    }
}

pub fn min_max(array: impl AsRef<Array>) -> VortexResult<MinMaxResult> {
    let array = array.as_ref();

    let min_max = if let Some(fn_) = array.vtable().min_max_fn() {
        fn_.min_max(array)?
    } else {
        let canonical = array.clone().into_canonical()?;
        if let Some(fn_) = canonical.vtable().min_max_fn() {
            fn_.min_max(canonical.as_ref())?
        } else {
            return Err(vortex_err!(NotImplemented: "min_max", array.encoding()));
        }
    };

    if let Some((min, max)) = &min_max {
        debug_assert_eq!(
            min.value().dtype(),
            array.dtype(),
            "MinMax min dtype mismatch {}",
            array.encoding()
        );
        debug_assert_eq!(
            max.value().dtype(),
            array.dtype(),
            "MinMax max dtype mismatch {}",
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

    use crate::array::{BoolArray, NullArray, PrimitiveArray};
    use crate::compute::min_max;
    use crate::stats::Precision;
    use crate::validity::Validity::NonNullable;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], NonNullable);
        assert_eq!(
            min_max(p).unwrap(),
            Some((Precision::exact(1), Precision::exact(3)))
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
            Some((Precision::exact(true), Precision::exact(true)))
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, false, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            Some((Precision::exact(false), Precision::exact(false)))
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, true, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            Some((Precision::exact(false), Precision::exact(true)))
        );
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(p).unwrap(), (None));
    }
}
