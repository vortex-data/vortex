// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::arrays::PrimitiveArray;
use vortex_dtype::{NativePType, Nullability, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_scalar::PValue;

use crate::SequenceArray;

pub fn sequence_encode(primitive_array: &PrimitiveArray) -> VortexResult<Option<ArrayRef>> {
    if primitive_array.is_empty() {
        // we cannot encode an empty array
        return Ok(None);
    }

    if !primitive_array.all_valid()? {
        return Ok(None);
    }

    if primitive_array.ptype().is_float() {
        // for now, we don't handle float arrays, due to possible precision issues
        return Ok(None);
    }

    match_each_integer_ptype!(primitive_array.ptype(), |P| {
        encode_primitive_array(
            primitive_array.as_slice::<P>(),
            primitive_array.dtype().nullability(),
        )
    })
}

fn encode_primitive_array<P: NativePType + Into<PValue>>(
    slice: &[P],
    nullability: Nullability,
) -> VortexResult<Option<ArrayRef>> {
    if slice.len() == 1 {
        // The multiplier here can be any value, zero is chosen
        return SequenceArray::typed_new(slice[0], P::zero(), nullability, 1)
            .map(|a| Some(a.to_array()));
    }
    let base = slice[0];
    let multiplier = slice[1] - slice[0];
    slice
        .windows(2)
        .all(|w| w[1] == w[0] + multiplier)
        .then_some(
            SequenceArray::typed_new(base, multiplier, nullability, slice.len())
                .map(|a| a.to_array()),
        )
        .transpose()
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;

    use crate::sequence_encode;

    #[test]
    fn test_encode_array_success() {
        let primitive_array = PrimitiveArray::from_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let encoded = sequence_encode(&primitive_array).unwrap();
        assert!(encoded.is_some());
        let decoded = encoded.unwrap().to_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i32>(), primitive_array.as_slice::<i32>());
    }

    #[test]
    fn test_encode_array_1_success() {
        let primitive_array = PrimitiveArray::from_iter([0]);
        let encoded = sequence_encode(&primitive_array).unwrap();
        assert!(encoded.is_some());
        let decoded = encoded.unwrap().to_primitive().unwrap();
        assert_eq!(decoded.as_slice::<i32>(), primitive_array.as_slice::<i32>());
    }

    #[test]
    fn test_encode_array_fail() {
        let primitive_array = PrimitiveArray::from_iter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);

        let encoded = sequence_encode(&primitive_array).unwrap();
        assert!(encoded.is_none());
    }
}
