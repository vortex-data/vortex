use arrow_buffer::BooleanBufferBuilder;
use itertools::Itertools;
use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_buffer::{buffer, Buffer, BufferMut};
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::iter::trimmed_ends_iter;

/// Run-end encode a `PrimitiveArray`, returning a tuple of `(ends, values)`.
pub fn runend_encode(array: &PrimitiveArray) -> VortexResult<(PrimitiveArray, Array)> {
    let validity = match array.validity() {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            return Ok((
                PrimitiveArray::new(buffer![array.len() as u64], Validity::NonNullable),
                ConstantArray::new(Scalar::null(array.dtype().clone()), 1).into_array(),
            ));
        }
        Validity::Array(a) => Some(a.into_bool()?.boolean_buffer()),
    };

    Ok(match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |$P| {
                let (ends, values) = runend_encode_primitive(array.as_slice::<$P>());
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    PrimitiveArray::new(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |$P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.as_slice::<$P>(), validity);
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    })
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Buffer<u64>, Buffer<T>) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();

    if elements.is_empty() {
        return (ends.freeze(), values.freeze());
    }

    // Run-end encode the values
    let mut prev = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != prev {
            ends.push(end);
            values.push(prev);
        }
        prev = e;
        end += 1;
    }
    ends.push(end);
    values.push(prev);

    (ends.freeze(), values.freeze())
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BooleanBuffer,
) -> (Buffer<u64>, PrimitiveArray) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();
    let mut validity = BooleanBufferBuilder::new(values.capacity());

    if elements.is_empty() {
        return (
            ends.freeze(),
            PrimitiveArray::new(
                values,
                Validity::Array(BoolArray::from(validity.finish()).into_array()),
            ),
        );
    }

    // Run-end encode the values
    let mut prev = element_validity.value(0).then(|| elements[0]);
    let mut end = 1;
    for e in elements
        .iter()
        .zip(element_validity.iter())
        .map(|(&e, is_valid)| is_valid.then_some(e))
        .skip(1)
    {
        if e != prev {
            ends.push(end);
            match prev {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(p) => {
                    validity.append(true);
                    values.push(p);
                }
            }
        }
        prev = e;
        end += 1;
    }
    ends.push(end);

    match prev {
        None => {
            validity.append(false);
            values.push(T::default());
        }
        Some(p) => {
            validity.append(true);
            values.push(p);
        }
    }

    (
        ends.freeze(),
        PrimitiveArray::new(values, Validity::from(validity.finish())),
    )
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    length: usize,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(values.ptype(), |$P| {
        match_each_integer_ptype!(ends.ptype(), |$E| {
            runend_decode_typed_primitive(
                trimmed_ends_iter(ends.as_slice::<$E>(), offset, length),
                values.as_slice::<$P>(),
                values.validity_mask()?,
                values.dtype().nullability(),
                length,
            )
        })
    })
}

pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> VortexResult<BoolArray> {
    match_each_integer_ptype!(ends.ptype(), |$E| {
        runend_decode_typed_bool(
            trimmed_ends_iter(ends.as_slice::<$E>(), offset, length),
            values.boolean_buffer(),
            values.validity_mask()?,
            values.dtype().nullability(),
            length,
        )
    })
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> VortexResult<PrimitiveArray> {
    Ok(match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
            for (end, value) in run_ends.zip_eq(values) {
                assert!(end <= length, "Runend end must be less than overall length");
                // SAFETY:
                // We preallocate enough capacity because we know the total length
                unsafe { decoded.push_n_unchecked(*value, end - decoded.len()) };
            }
            PrimitiveArray::new(decoded, values_nullability.into())
        }
        Mask::AllFalse(_) => PrimitiveArray::new(Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            let mut decoded = BufferMut::with_capacity(length);
            let mut decoded_validity = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(mask.boolean_buffer().iter())
                    .map(|(&v, is_valid)| is_valid.then_some(v)),
            ) {
                assert!(end <= length, "Runend end must be less than overall length");
                match value {
                    None => {
                        decoded_validity.append_n(end - decoded.len(), false);
                        // SAFETY:
                        // We preallocate enough capacity because we know the total length
                        unsafe { decoded.push_n_unchecked(T::default(), end - decoded.len()) };
                    }
                    Some(value) => {
                        decoded_validity.append_n(end - decoded.len(), true);
                        // SAFETY:
                        // We preallocate enough capacity because we know the total length
                        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
                    }
                }
            }
            PrimitiveArray::new(decoded, Validity::from(decoded_validity.finish()))
        }
    })
}

pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: BooleanBuffer,
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> VortexResult<BoolArray> {
    Ok(match values_validity {
        Mask::AllTrue(_) => {
            let mut decoded = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(values.iter()) {
                decoded.append_n(end - decoded.len(), value);
            }
            BoolArray::new(decoded.finish(), values_nullability)
        }
        Mask::AllFalse(_) => {
            BoolArray::try_new(BooleanBuffer::new_unset(length), Validity::AllInvalid)
                .vortex_expect("invalid array")
        }
        Mask::Values(mask) => {
            let mut decoded = BooleanBufferBuilder::new(length);
            let mut decoded_validity = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(mask.boolean_buffer().iter())
                    .map(|(v, is_valid)| is_valid.then_some(v)),
            ) {
                match value {
                    None => {
                        decoded_validity.append_n(end - decoded.len(), false);
                        decoded.append_n(end - decoded.len(), false);
                    }
                    Some(value) => {
                        decoded_validity.append_n(end - decoded.len(), true);
                        decoded.append_n(end - decoded.len(), value);
                    }
                }
            }
            BoolArray::try_new(decoded.finish(), Validity::from(decoded_validity.finish()))?
        }
    })
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;
    use vortex_buffer::buffer;

    use crate::compress::{runend_decode_primitive, runend_encode};

    #[test]
    fn encode() {
        let arr = PrimitiveArray::from_iter([1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(&arr).unwrap();
        let values = values.into_primitive().unwrap();

        assert_eq!(ends.as_slice::<u64>(), vec![2, 5, 10]);
        assert_eq!(values.as_slice::<i32>(), vec![1, 2, 3]);
    }

    #[test]
    fn encode_nullable() {
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BooleanBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(&arr).unwrap();
        let values = values.into_primitive().unwrap();

        assert_eq!(ends.as_slice::<u64>(), vec![2, 4, 5, 8, 10]);
        assert_eq!(values.as_slice::<i32>(), vec![1, 0, 2, 3, 0]);
    }

    #[test]
    fn encode_all_null() {
        let arr = PrimitiveArray::new(
            buffer![0, 0, 0, 0, 0],
            Validity::from(BooleanBuffer::new_unset(5)),
        );
        let (ends, values) = runend_encode(&arr).unwrap();
        let values = values.into_primitive().unwrap();

        assert_eq!(ends.as_slice::<u64>(), vec![5]);
        assert_eq!(values.as_slice::<i32>(), vec![0]);
    }

    #[test]
    fn decode() {
        let ends = PrimitiveArray::from_iter([2, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10).unwrap();

        assert_eq!(
            decoded.as_slice::<i32>(),
            vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }
}
