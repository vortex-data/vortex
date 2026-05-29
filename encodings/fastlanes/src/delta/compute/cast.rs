// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::delta::Delta;
use crate::delta::array::DeltaArrayExt;

impl CastReduce for Delta {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(target_ptype, _) = dtype else {
            return Ok(None);
        };

        let source_ptype = array.dtype().as_ptype();
        // Only a same-width cast (e.g. a nullability change) can be served by
        // re-casting the stored components in place. Any width change must defer
        // to the decompress-then-cast fallback (`Ok(None)`):
        //
        // * Widening cannot be done in place. `bases`/`deltas` are held in
        //   FastLanes transposed layout with `T::LANES` (= 1024 / bit_width)
        //   entries per chunk, and `T::LANES` changes with the target width.
        //   Re-widening the buffers element-wise preserves the *source* width's
        //   layout, but `delta_decompress` then reads them with the *target*
        //   width's lane count, decoding against a misaligned layout and
        //   producing wrong (and, for `onpair` dictionary offsets, non-monotonic)
        //   values for any array larger than a single near-empty chunk.
        // * Narrowing is unsafe without first decompressing to check the max
        //   value fits.
        if source_ptype.bit_width() != target_ptype.bit_width() {
            return Ok(None);
        }
        // Signed sources need a different cast policy than the lossless cast used
        // here. The delta bytes are stored as the result of `wrapping_sub`, so e.g.
        // a delta of -1i8 has the bit pattern 0xFF. Widening *as a value* (the cast op's
        // semantics) sign-extends that to 0xFFFFFFFF, which means `wrapping_add(base, delta)`
        // at the wider type produces a different result than at the source type — round-trip
        // breaks. Cross-signedness widening has the same hazard for the same reason. Fall
        // back to decompress-and-re-encode for both cases.
        if target_ptype.is_signed_int() || source_ptype.is_signed_int() {
            return Ok(None);
        }

        // Cast both bases and deltas to the target type
        let casted_bases = array.bases().cast(dtype.with_nullability(NonNullable))?;
        let casted_deltas = array.deltas().cast(dtype.clone())?;

        // Create a new DeltaArray with the casted components, preserving offset and logical length
        Ok(Some(
            Delta::try_new(casted_bases, casted_deltas, array.offset(), array.len())?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::Delta;
    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_cast_delta_u8_to_u32() {
        let primitive = PrimitiveArray::from_iter([10u8, 20, 30, 40, 50]);
        let array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        // Verify by decoding
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([10u32, 20, 30, 40, 50]));
    }

    /// Widening across more than one FastLanes chunk (len > 1024). The in-place
    /// component cast is invalid here because `T::LANES` differs between source
    /// and target widths, so this must fall back to decompress-then-cast. A
    /// previous in-place widen produced non-monotonic values and corrupted
    /// round-trips (the `onpair` dictionary-offsets panic).
    #[rstest]
    #[case::u8_to_u32(8)]
    #[case::u16_to_u32(16)]
    fn test_cast_delta_widen_multichunk(#[case] src_width: u32) {
        let n = 4096usize;
        let expected: Vec<u32> = (0..n as u32).map(|i| (i * 3) % 60_000).collect();
        let delta = match src_width {
            8 => Delta::try_from_primitive_array(
                &PrimitiveArray::from_iter((0..n).map(|i| ((i * 3) % 250) as u8)),
                &mut SESSION.create_execution_ctx(),
            ),
            _ => Delta::try_from_primitive_array(
                &PrimitiveArray::from_iter(expected.iter().map(|&v| v as u16)),
                &mut SESSION.create_execution_ctx(),
            ),
        }
        .unwrap();
        let expected: Vec<u32> = if src_width == 8 {
            (0..n).map(|i| ((i * 3) % 250) as u32).collect()
        } else {
            expected
        };

        let casted = delta
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(casted.as_slice::<u32>(), expected.as_slice());
    }

    #[test]
    fn test_cast_delta_nullable() {
        // DeltaArray doesn't support nullable arrays - the validity is handled at the DeltaArray level
        // Create a non-nullable array and then add validity to the DeltaArray
        let values = PrimitiveArray::new(
            buffer![100u16, 0, 200, 300, 0],
            vortex_array::validity::Validity::NonNullable,
        );
        let array =
            Delta::try_from_primitive_array(&values, &mut SESSION.create_execution_ctx()).unwrap();

        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case::u8(
        PrimitiveArray::new(
            buffer![0u8, 10, 20, 30, 40, 50],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u16(
        PrimitiveArray::new(
            buffer![0u16, 100, 200, 300, 400, 500],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u32(
        PrimitiveArray::new(
            buffer![0u32, 1000, 2000, 3000, 4000],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    #[case::u64(
        PrimitiveArray::new(
            buffer![0u64, 10000, 20000, 30000],
            vortex_array::validity::Validity::NonNullable,
        )
    )]
    fn test_cast_delta_conformance(#[case] primitive: PrimitiveArray) {
        let delta_array =
            Delta::try_from_primitive_array(&primitive, &mut SESSION.create_execution_ctx())
                .unwrap();
        test_cast_conformance(&delta_array.into_array());
    }
}
