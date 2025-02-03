use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::stats::{Precision, Stat, Statistics};
use crate::{Array, Encoding, IntoCanonical};

pub type MinMaxResult = (Option<Scalar>, Option<Scalar>);

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

    if let (Some(min), _) = &min_max {
        debug_assert_eq!(
            min.dtype(),
            array.dtype(),
            "MinMax min dtype mismatch {}",
            array.encoding()
        );

        array.set(Stat::Min, Precision::exact(min.clone().into_value()));
    }

    if let (_, Some(max)) = &min_max {
        debug_assert_eq!(
            max.dtype(),
            array.dtype(),
            "MinMax min dtype mismatch {}",
            array.encoding()
        );
        array.set(Stat::Max, Precision::exact(max.clone().into_value()));
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
    use crate::validity::Validity::NonNullable;

    #[test]
    fn test_prim_max() {
        let p = PrimitiveArray::new(buffer![1, 2, 3], NonNullable);
        assert_eq!(min_max(p).unwrap(), (Some(1.into()), Some(3.into())));
    }

    #[test]
    fn test_bool_max() {
        let p = BoolArray::new(
            BooleanBuffer::from([true, true, true].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(min_max(p).unwrap(), (Some(true.into()), Some(true.into())));

        let p = BoolArray::new(
            BooleanBuffer::from([false, false, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(
            min_max(p).unwrap(),
            (Some(false.into()), Some(false.into()))
        );

        let p = BoolArray::new(
            BooleanBuffer::from([false, true, false].as_slice()),
            Nullability::NonNullable,
        );
        assert_eq!(min_max(p).unwrap(), (Some(false.into()), Some(true.into())));
    }

    #[test]
    fn test_null() {
        let p = NullArray::new(1);
        assert_eq!(min_max(p).unwrap(), (None, None));
    }
}
