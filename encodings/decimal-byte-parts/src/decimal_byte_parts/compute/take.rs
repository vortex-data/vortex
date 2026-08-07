// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use crate::decimal_byte_parts::map_parts;

impl TakeReduce for DecimalByteParts {
    /// Taking wraps each part in a `Dict` without reading any buffer, so it reduces rather
    /// than executes.
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Taking with nullable indices makes every taken part nullable, but lower parts must
        // stay non-nullable `u64` — validity belongs to the MSP alone. Fall back to the
        // canonical path rather than rebuilding parts we would have to strip nullability from.
        if indices.dtype().is_nullable() && !array.lower_parts().is_empty() {
            return Ok(None);
        }

        map_parts(array, |part| part.take(indices.clone())).map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;
    use crate::decimal_byte_parts::testing::encode;
    use crate::decimal_byte_parts::testing::i256_of;

    /// Taking pushes down into the parts during optimization, with no execution context in
    /// play: `ArrayRef::take` wraps the array in a `Dict` and optimizes, and the reduce rule
    /// must rewrite that into a `DecimalByteParts` of taken parts.
    #[test]
    fn take_pushes_down_without_executing() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);

        let decimal = DecimalArray::new(
            Buffer::from(vec![1i128 << 70, 2, 3]),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let indices = buffer![0u64, 2].into_array();
        let taken = encode(&decimal)?.into_array().take(indices)?;

        assert!(
            taken.is::<DecimalByteParts>(),
            "expected the take to reduce into the encoding, got {}",
            taken.encoding_id()
        );
        Ok(())
    }

    /// Taking with nullable indices must still round-trip the wide values, including the
    /// null row, on arrays that carry lower parts.
    #[rstest]
    #[case::one_lower_part(DecimalArray::new(
        Buffer::from(vec![1i128 << 70, 2, 3]),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    ))]
    #[case::three_lower_parts(DecimalArray::new(
        Buffer::from(vec![i256_of(1, 1 << 70), i256_of(0, 2), i256_of(0, 3)]),
        DecimalDType::new(76, 2),
        Validity::NonNullable,
    ))]
    fn take_with_nullable_indices(#[case] decimal: DecimalArray) -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();
        let expected = decimal
            .clone()
            .into_array()
            .take(indices.clone())?
            .execute::<DecimalArray>(&mut ctx)?;

        let taken = encode(&decimal)?.into_array().take(indices)?;
        let actual = taken.execute::<DecimalArray>(&mut ctx)?;

        assert_arrays_eq!(expected, actual, &mut ctx);
        Ok(())
    }
}
