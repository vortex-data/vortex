use arrow_buffer::BooleanBufferBuilder;
use itertools::Itertools;
use vortex_array::array::{BoolArray, BooleanBuffer, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::ArrayDType;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType, Nullability};
use vortex_error::VortexResult;

use crate::iter::trimmed_ends_iter;

pub fn runend_encode(array: &PrimitiveArray) -> (PrimitiveArray, PrimitiveArray) {
    let validity = if array.dtype().nullability() == Nullability::NonNullable {
        Validity::NonNullable
    } else {
        Validity::AllValid
    };

    let (compressed_ends, compressed_values) = match_each_native_ptype!(array.ptype(), |$P| {
        let (ends, values) = runend_encode_primitive(array.maybe_null_slice::<$P>());
        (PrimitiveArray::from_vec(ends, Validity::NonNullable), PrimitiveArray::from_vec(values, validity))
    });

    assert_eq!(array.dtype(), compressed_values.dtype());
    (compressed_ends, compressed_values)
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Vec<u64>, Vec<T>) {
    let mut ends = Vec::new();
    let mut values = Vec::new();

    if elements.is_empty() {
        return (ends, values);
    }

    // Run-end encode the values
    let mut last = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != last {
            ends.push(end);
            values.push(last);
        }
        last = e;
        end += 1;
    }
    ends.push(end);
    values.push(last);

    (ends, values)
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    validity: Validity,
    offset: usize,
    length: usize,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(values.ptype(), |$P| {
        match_each_integer_ptype!(ends.ptype(), |$E| {
            Ok(PrimitiveArray::from_vec(runend_decode_typed_primitive(
                trimmed_ends_iter(ends.maybe_null_slice::<$E>(), offset, length),
                values.maybe_null_slice::<$P>(),
                length,
            ), validity))
        })
    })
}

pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    validity: Validity,
    offset: usize,
    length: usize,
) -> VortexResult<BoolArray> {
    match_each_integer_ptype!(ends.ptype(), |$E| {
        BoolArray::try_new(runend_decode_typed_bool(
            trimmed_ends_iter(ends.maybe_null_slice::<$E>(), offset, length),
            values.boolean_buffer(),
            length,
        ), validity)
    })
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    length: usize,
) -> Vec<T> {
    let mut decoded = Vec::with_capacity(length);
    for (end, value) in run_ends.zip_eq(values) {
        decoded.extend(std::iter::repeat_n(value, end - decoded.len()));
    }
    decoded
}

pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: BooleanBuffer,
    length: usize,
) -> BooleanBuffer {
    let mut decoded = BooleanBufferBuilder::new(length);
    for (end, value) in run_ends.zip_eq(values.iter()) {
        decoded.append_n(end - decoded.len(), value);
    }
    decoded.finish()
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::{ArrayValidity, Validity};
    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};

    use crate::compress::{runend_decode_primitive, runend_encode};
    use crate::RunEndArray;

    #[test]
    fn encode() {
        let arr = PrimitiveArray::from(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(&arr);

        assert_eq!(ends.maybe_null_slice::<u64>(), vec![2, 5, 10]);
        assert_eq!(values.maybe_null_slice::<i32>(), vec![1, 2, 3]);
    }

    #[test]
    fn decode() {
        let ends = PrimitiveArray::from(vec![2, 5, 10]);
        let values = PrimitiveArray::from(vec![1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, Validity::NonNullable, 0, 10).unwrap();

        assert_eq!(
            decoded.maybe_null_slice::<i32>(),
            vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn decode_nullable() {
        let validity = {
            let mut validity = vec![true; 10];
            validity[2] = false;
            validity[7] = false;
            Validity::from_iter(validity)
        };
        let arr = RunEndArray::try_new(
            vec![2u32, 5, 10].into_array(),
            PrimitiveArray::from_vec(vec![1i32, 2, 3], Validity::AllValid).into_array(),
            validity,
        )
        .unwrap();

        let decoded = runend_decode_primitive(
            arr.ends().into_primitive().unwrap(),
            arr.values().into_primitive().unwrap(),
            arr.validity(),
            0,
            arr.len(),
        )
        .unwrap();

        assert_eq!(
            decoded.maybe_null_slice::<i32>(),
            vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3].as_slice()
        );
        assert_eq!(
            decoded.logical_validity().into_validity(),
            Validity::from_iter([true, true, false, true, true, true, true, false, true, true,])
        );
    }
}
