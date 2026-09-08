// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::array::DeltaArraySlotsExt;

impl ValidityVTable<Delta> for Delta {
    fn validity(array: ArrayView<'_, Delta>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.validity_child(),
            array.dtype().nullability(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SliceArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Alignment;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use super::*;
    use crate::delta::array::delta_compress::delta_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Length not a multiple of 1,024, so the last chunk is padded.
    const LEN: usize = 2500;

    fn nullable_primitive() -> PrimitiveArray {
        PrimitiveArray::from_option_iter(
            (0u32..LEN as u32).map(|value| (value % 3 != 0).then_some(value)),
        )
    }

    fn expected_validity(range: Range<usize>) -> BoolArray {
        BoolArray::from_iter(range.map(|value| value % 3 != 0))
    }

    /// Rebuilds `bits` one byte into a fresh allocation so the backing buffer is not u64-aligned,
    /// like a bitmap viewed out of a file segment.
    fn misalign_bits(bits: BitBuffer) -> BitBuffer {
        let (_offset, len, bytes) = bits.sliced().into_inner();
        let mut shifted = ByteBufferMut::with_capacity(bytes.len() + 1);
        shifted.push(0xFF);
        shifted.extend_from_slice(&bytes);
        let misaligned = shifted.freeze().slice(1..bytes.len() + 1);
        assert!(!misaligned.is_aligned(Alignment::of::<u64>()));
        BitBuffer::new(misaligned, len)
    }

    #[test]
    fn validity_is_lazy_for_cross_chunk_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = nullable_primitive();
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        let sliced = delta.slice(1000..1050)?;

        let Validity::Array(validity) = sliced.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(validity, expected_validity(1000..1050), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::within_first_chunk(10..1000)]
    #[case::cross_chunk_boundary(1000..1050)]
    #[case::chunk_aligned(1024..2048)]
    #[case::tail_into_padded_chunk(2400..LEN)]
    #[case::full_range(0..LEN)]
    fn sliced_validity_matches_slice_range(#[case] range: Range<usize>) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = nullable_primitive();
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        let sliced = delta.slice(range.clone())?;

        let Validity::Array(validity) = sliced.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(validity, expected_validity(range.clone()), &mut ctx);
        assert_arrays_eq!(sliced, primitive.slice(range)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn validity_of_nested_slice_composes_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = nullable_primitive();
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;
        let sliced = delta.slice(100..2400)?.slice(900..1500)?;

        let Validity::Array(validity) = sliced.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(validity, expected_validity(1000..1600), &mut ctx);
        assert_arrays_eq!(sliced, primitive.slice(1000..1600)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn validity_handles_slice_encoding() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0u32..2048).map(|value| (value % 3 != 0).then_some(value)),
        );
        let (bases, deltas) = delta_compress(&primitive, &mut ctx)?;

        let Validity::Array(validity) = primitive.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        let lazy_validity = SliceArray::try_new(validity, 0..primitive.len())?.into_array();
        let delta = Delta::try_new(
            bases.into_array(),
            deltas.into_array(),
            Validity::Array(lazy_validity),
            0,
            primitive.len(),
        )?;

        let Validity::Array(validity) = delta.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(
            validity,
            BoolArray::from_iter((0u32..2048).map(|value| value % 3 != 0)),
            &mut ctx
        );
        assert_arrays_eq!(delta, primitive, &mut ctx);
        Ok(())
    }

    #[test]
    fn validity_from_unaligned_buffer() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = nullable_primitive();
        let (bases, deltas) = delta_compress(&primitive, &mut ctx)?;

        let Validity::Array(validity) = primitive.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        let bits = validity.execute::<BoolArray>(&mut ctx)?.into_bit_buffer();
        let unaligned = BoolArray::new(misalign_bits(bits), Validity::NonNullable).into_array();
        let delta = Delta::try_new(
            bases.into_array(),
            deltas.into_array(),
            Validity::Array(unaligned),
            0,
            primitive.len(),
        )?;

        let Validity::Array(validity) = delta.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(validity, expected_validity(0..LEN), &mut ctx);
        assert_arrays_eq!(delta, primitive, &mut ctx);

        let sliced = delta.slice(1000..1050)?;
        let Validity::Array(sliced_validity) = sliced.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(sliced_validity, expected_validity(1000..1050), &mut ctx);
        Ok(())
    }

    #[test]
    fn compress_primitive_with_unaligned_validity_buffer() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 2048u32;
        let bits = misalign_bits(BitBuffer::from_iter((0..len).map(|value| value % 3 != 0)));
        let mask = BoolArray::new(bits, Validity::NonNullable).into_array();
        let primitive =
            PrimitiveArray::new((0..len).collect::<Buffer<u32>>(), Validity::Array(mask));
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?;

        let Validity::Array(validity) = delta.validity()? else {
            vortex_bail!("expected array-backed validity")
        };
        assert_arrays_eq!(validity, expected_validity(0..len as usize), &mut ctx);
        assert_arrays_eq!(delta, primitive, &mut ctx);
        Ok(())
    }
}
