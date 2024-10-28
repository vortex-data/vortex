use std::cmp::min;

use itertools::Itertools;
use num_traits::{AsPrimitive, FromPrimitive};
use vortex_array::array::PrimitiveArray;
use vortex_array::compute::unary::scalar_at;
use vortex_array::stats::{ArrayStatistics, Stat};
use vortex_array::validity::Validity;
use vortex_array::ArrayDType;
use vortex_dtype::{match_each_integer_ptype, match_each_native_ptype, NativePType, Nullability};
use vortex_error::{vortex_panic, VortexResult, VortexUnwrap as _};

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

    // the values array stats are trivially derived
    compressed_values
        .statistics()
        .set(Stat::RunCount, compressed_values.len().into());
    compressed_values
        .statistics()
        .set(Stat::IsConstant, (compressed_values.len() == 1).into());
    if let Some(min) = array.statistics().get(Stat::Min) {
        compressed_values.statistics().set(Stat::Min, min);
    }
    if let Some(max) = array.statistics().get(Stat::Max) {
        compressed_values.statistics().set(Stat::Max, max);
    }
    if let Some(is_sorted) = array.statistics().get(Stat::IsSorted) {
        compressed_values
            .statistics()
            .set(Stat::IsSorted, is_sorted);
    }
    if let Some(is_strict_sorted) = array.statistics().get(Stat::IsStrictSorted) {
        compressed_values
            .statistics()
            .set(Stat::IsStrictSorted, is_strict_sorted);
    }

    compressed_ends
        .statistics()
        .set(Stat::IsConstant, (compressed_ends.len() == 1).into());
    compressed_ends
        .statistics()
        .set(Stat::IsSorted, true.into());
    compressed_ends
        .statistics()
        .set(Stat::IsStrictSorted, true.into());
    if !compressed_ends.is_empty() {
        compressed_ends
            .statistics()
            .set(Stat::Min, scalar_at(&compressed_ends, 0).vortex_unwrap());
        compressed_ends
            .statistics()
            .set(Stat::Max, (array.len() as u64).into());
    }

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

pub fn runend_decode(
    ends: &PrimitiveArray,
    values: &PrimitiveArray,
    validity: Validity,
    offset: usize,
    length: usize,
) -> VortexResult<PrimitiveArray> {
    match_each_native_ptype!(values.ptype(), |$P| {
        match_each_integer_ptype!(ends.ptype(), |$E| {
            Ok(PrimitiveArray::from_vec(runend_decode_primitive(
                ends.maybe_null_slice::<$E>(),
                values.maybe_null_slice::<$P>(),
                offset,
                length,
            ), validity))
        })
    })
}

pub fn runend_decode_primitive<
    E: NativePType + AsPrimitive<usize> + FromPrimitive + Ord,
    T: NativePType,
>(
    run_ends: &[E],
    values: &[T],
    offset: usize,
    length: usize,
) -> Vec<T> {
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });
    let trimmed_ends = run_ends
        .iter()
        .map(|v| *v - offset_e)
        .map(|v| min(v, length_e));

    let mut decoded = Vec::with_capacity(length);
    for (end, &value) in trimmed_ends.zip_eq(values) {
        decoded.extend(std::iter::repeat(value).take(end.as_() - decoded.len()));
    }
    decoded
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::{ArrayValidity, Validity};
    use vortex_array::IntoArray;

    use crate::compress::{runend_decode, runend_encode};
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
        let decoded = runend_decode(&ends, &values, Validity::NonNullable, 0, 10).unwrap();

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
            Validity::from(validity)
        };
        let arr = RunEndArray::try_new(
            vec![2u32, 5, 10].into_array(),
            PrimitiveArray::from_vec(vec![1i32, 2, 3], Validity::AllValid).into_array(),
            validity,
        )
        .unwrap();

        let decoded = runend_decode(
            &arr.ends().as_primitive(),
            &arr.values().as_primitive(),
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
            Validity::from(vec![
                true, true, false, true, true, true, true, false, true, true,
            ])
        );
    }
}
