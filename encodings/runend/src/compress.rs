use arrow_buffer::BooleanBufferBuilder;
use itertools::Itertools;
use vortex_array::array::{BoolArray, BooleanBuffer, ConstantArray, PrimitiveArray};
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::iter::trimmed_ends_iter;

pub fn runend_encode(array: &PrimitiveArray) -> VortexResult<(PrimitiveArray, ArrayData)> {
    let validity = match array.validity() {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            return Ok((
                PrimitiveArray::from(vec![array.len() as u64]),
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            ));
        }
        Validity::Array(a) => Some(a.into_bool()?.boolean_buffer()),
    };

    Ok(match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |$P| {
                let (ends, values) = runend_encode_primitive(array.maybe_null_slice::<$P>());
                (
                    PrimitiveArray::from_vec(ends, Validity::NonNullable),
                    PrimitiveArray::from_vec(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |$P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.maybe_null_slice::<$P>(), validity);
                (
                    PrimitiveArray::from_vec(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    })
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Vec<u64>, Vec<T>) {
    let mut ends = Vec::new();
    let mut values = Vec::new();

    if elements.is_empty() {
        return (ends, values);
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

    (ends, values)
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BooleanBuffer,
) -> (Vec<u64>, PrimitiveArray) {
    let mut ends = Vec::new();
    let mut values = Vec::new();
    let mut validity = BooleanBufferBuilder::new(values.capacity());

    if elements.is_empty() {
        return (
            ends,
            PrimitiveArray::from_vec(
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
        ends,
        PrimitiveArray::from_vec(
            values,
            Validity::Array(BoolArray::from(validity.finish()).into_array()),
        ),
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
                trimmed_ends_iter(ends.maybe_null_slice::<$E>(), offset, length),
                values.maybe_null_slice::<$P>(),
                values.logical_validity(),
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
            trimmed_ends_iter(ends.maybe_null_slice::<$E>(), offset, length),
            values.boolean_buffer(),
            values.logical_validity(),
            values.dtype().nullability(),
            length,
        )
    })
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: LogicalValidity,
    values_nullability: Nullability,
    length: usize,
) -> VortexResult<PrimitiveArray> {
    Ok(match values_validity {
        LogicalValidity::AllValid(_) => {
            let mut decoded: Vec<T> = Vec::with_capacity(length);
            for (end, value) in run_ends.zip_eq(values) {
                decoded.extend(std::iter::repeat_n(value, end - decoded.len()));
            }
            PrimitiveArray::from_vec(decoded, values_nullability.into())
        }
        LogicalValidity::AllInvalid(_) => PrimitiveArray::from_vec(
            vec![T::default(); length],
            Validity::Array(BoolArray::from(BooleanBuffer::new_unset(length)).into_array()),
        ),
        LogicalValidity::Array(array) => {
            let validity = array.into_bool()?.boolean_buffer();
            let mut decoded = Vec::with_capacity(length);
            let mut decoded_validity = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(validity.iter())
                    .map(|(&v, is_valid)| is_valid.then_some(v)),
            ) {
                match value {
                    None => {
                        decoded_validity.append_n(end - decoded.len(), false);
                        decoded.extend(std::iter::repeat_n(T::default(), end - decoded.len()));
                    }
                    Some(value) => {
                        decoded_validity.append_n(end - decoded.len(), true);
                        decoded.extend(std::iter::repeat_n(value, end - decoded.len()));
                    }
                }
            }
            PrimitiveArray::from_vec(
                decoded,
                Validity::Array(BoolArray::from(decoded_validity.finish()).into_array()),
            )
        }
    })
}

pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: BooleanBuffer,
    values_validity: LogicalValidity,
    values_nullability: Nullability,
    length: usize,
) -> VortexResult<BoolArray> {
    Ok(match values_validity {
        LogicalValidity::AllValid(_) => {
            let mut decoded = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(values.iter()) {
                decoded.append_n(end - decoded.len(), value);
            }
            BoolArray::new(decoded.finish(), values_nullability)
        }
        LogicalValidity::AllInvalid(_) => BoolArray::try_new(
            BooleanBuffer::new_unset(length),
            Validity::Array(BoolArray::from(BooleanBuffer::new_unset(length)).into_array()),
        )
        .vortex_expect("invalid array"),
        LogicalValidity::Array(array) => {
            let validity = array.into_bool()?.boolean_buffer();
            let mut decoded = BooleanBufferBuilder::new(length);
            let mut decoded_validity = BooleanBufferBuilder::new(length);
            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(validity.iter())
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
            BoolArray::try_new(
                decoded.finish(),
                Validity::Array(BoolArray::from(decoded_validity.finish()).into_array()),
            )?
        }
    })
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;

    use crate::compress::{runend_decode_primitive, runend_encode};

    #[test]
    fn encode() {
        let arr = PrimitiveArray::from(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(&arr).unwrap();
        let values = values.into_primitive().unwrap();

        assert_eq!(ends.maybe_null_slice::<u64>(), vec![2, 5, 10]);
        assert_eq!(values.maybe_null_slice::<i32>(), vec![1, 2, 3]);
    }

    #[test]
    fn encode_nullable() {
        let arr = PrimitiveArray::from_vec(
            vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BooleanBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(&arr).unwrap();
        let values = values.into_primitive().unwrap();

        assert_eq!(ends.maybe_null_slice::<u64>(), vec![2, 4, 5, 8, 10]);
        assert_eq!(values.maybe_null_slice::<i32>(), vec![1, 0, 2, 3, 0]);
    }

    #[test]
    fn decode() {
        let ends = PrimitiveArray::from(vec![2, 5, 10]);
        let values = PrimitiveArray::from(vec![1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10).unwrap();

        assert_eq!(
            decoded.maybe_null_slice::<i32>(),
            vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }
}
