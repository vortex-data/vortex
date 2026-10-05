// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the global bit width and block offsets of bit-packed arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedData;
use crate::BitWidths;
use crate::BitWidthsView;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn encode(values: &[u32]) -> VortexResult<BitPackedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    bitpack_to_best_bit_width(&PrimitiveArray::from_iter(values.iter().copied()), &mut ctx)
}

fn uniform() -> VortexResult<BitPackedArray> {
    encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())
}

fn with_block_offsets(array: &BitPackedArray, offsets: ArrayRef) -> VortexResult<BitPackedArray> {
    BitPacked::try_new_with_block_offsets(
        array.packed().clone(),
        array.dtype().as_ptype(),
        array.validity()?,
        array.patches(),
        offsets,
        array.len(),
        array.offset(),
    )
}

#[test]
fn global_bit_width_has_no_block_offsets() -> VortexResult<()> {
    let uniform = uniform()?;
    assert!(matches!(uniform.bit_widths(), BitWidthsView::Global(7)));
    assert!(uniform.block_offsets().is_none());
    Ok(())
}

#[rstest]
#[case::both(true, true)]
#[case::neither(false, false)]
fn global_bit_width_and_block_offsets_are_exclusive(
    #[case] constant: bool,
    #[case] offsets: bool,
) -> VortexResult<()> {
    let packed = BufferHandle::new_host(ByteBuffer::zeroed(128));
    let data = if constant {
        BitPackedData::try_new(packed, None, 1, 0)?
    } else {
        BitPackedData::try_new_blocked(packed, None, 0)?
    };
    let block_offsets = offsets.then(|| buffer![0u64, 128].into_array());
    let slots: ArraySlots = [None, None, None, None, block_offsets]
        .into_iter()
        .collect();
    let dtype = DType::Primitive(PType::U32, Nullability::NonNullable);
    assert!(
        Array::<BitPacked>::try_from_parts(ArrayParts::new(BitPacked, dtype, 1024, data, slots))
            .is_err()
    );
    Ok(())
}

#[rstest]
fn unsigned_block_offsets_are_supported(
    #[values(PType::U8, PType::U16, PType::U32, PType::U64)] ptype: PType,
) -> VortexResult<()> {
    let offsets = match_each_unsigned_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter([127 as T, 255 as T]).into_array()
    });
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(128)),
        PType::U32,
        Validity::NonNullable,
        None,
        offsets.clone(),
        1024,
        0,
    )?;
    assert!(
        array
            .block_offsets()
            .is_some_and(|block_offsets| ArrayRef::ptr_eq(&offsets, block_offsets))
    );
    Ok(())
}

#[cfg(debug_assertions)]
#[rstest]
#[case::unaligned([0, 127])]
#[case::decreasing([128, 0])]
#[case::wrong_span([0, 0])]
#[should_panic(expected = "invalid BitPacked block offsets")]
fn invalid_unsigned_block_boundaries_panic_in_debug(
    #[case] boundaries: [u8; 2],
    #[values(PType::U8, PType::U16, PType::U32, PType::U64)] ptype: PType,
) {
    let offsets = match_each_unsigned_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter(boundaries.map(T::from)).into_array()
    });
    BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(128)),
        PType::U32,
        Validity::NonNullable,
        None,
        offsets,
        1024,
        0,
    )
    .unwrap();
}

#[rstest]
#[case::equal_steps(buffer![0u64, 896, 1792, 2688])]
#[case::different_widths(buffer![0u64, 384, 1408, 2688])]
fn block_offsets_have_no_constant_width(
    #[case] offsets: vortex_buffer::Buffer<u64>,
) -> VortexResult<()> {
    let array = with_block_offsets(&uniform()?, offsets.into_array())?;
    assert!(matches!(array.bit_widths(), BitWidthsView::Blocked(_)));
    // Decoding per-block widths is not supported yet.
    assert!(
        array
            .into_array()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .is_err()
    );
    Ok(())
}

#[rstest]
#[case::equal_steps(buffer![0u64, 896, 1792, 2688])]
#[case::different_widths(buffer![0u64, 384, 1408, 2688])]
fn casts_with_block_offsets_decline(
    #[case] offsets: vortex_buffer::Buffer<u64>,
    #[values(
        DType::Primitive(PType::U32, Nullability::Nullable),
        DType::Primitive(PType::U64, Nullability::NonNullable)
    )]
    dtype: DType,
) -> VortexResult<()> {
    let array = with_block_offsets(&uniform()?, offsets.into_array())?;
    let mut ctx = SESSION.create_execution_ctx();

    assert!(<BitPacked as CastReduce>::cast(array.as_view(), &dtype)?.is_none());
    assert!(<BitPacked as CastKernel>::cast(array.as_view(), &dtype, &mut ctx)?.is_none());
    Ok(())
}

#[rstest]
#[case::too_few_boundaries(buffer![0u64, 896, 1792].into_array())]
#[case::signed(buffer![0i32, 896, 1792, 2688].into_array())]
#[case::float(buffer![0f32, 896.0, 1792.0, 2688.0].into_array())]
#[case::nullable(
    PrimitiveArray::from_option_iter([Some(0u32), Some(896), Some(1792), Some(2688)]).into_array()
)]
fn invalid_block_offsets_are_rejected(#[case] offsets: ArrayRef) -> VortexResult<()> {
    assert!(with_block_offsets(&uniform()?, offsets).is_err());
    Ok(())
}

#[cfg(debug_assertions)]
#[rstest]
#[case::unaligned_block(buffer![0u64, 896, 1791, 2688])]
#[case::decreasing(buffer![0u64, 896, 768, 2688])]
#[case::span_disagrees_with_packed_len(buffer![0u64, 768, 1536, 2304])]
#[should_panic(expected = "invalid BitPacked block offsets")]
fn invalid_block_boundaries_panic_in_debug(#[case] offsets: vortex_buffer::Buffer<u64>) {
    with_block_offsets(&uniform().unwrap(), offsets.into_array()).unwrap();
}

/// One block of `ptype` values packed at `bit_width`, either globally or through block offsets.
fn single_block(ptype: PType, bit_width: u8, block_offsets: bool) -> VortexResult<BitPackedArray> {
    let packed = BufferHandle::new_host(ByteBuffer::zeroed(128 * usize::from(bit_width)));
    if block_offsets {
        let end = 128 * u64::from(bit_width);
        BitPacked::try_new_with_block_offsets(
            packed,
            ptype,
            Validity::NonNullable,
            None,
            buffer![0u64, end].into_array(),
            1024,
            0,
        )
    } else {
        BitPacked::try_new(
            packed,
            ptype,
            Validity::NonNullable,
            None,
            bit_width,
            1024,
            0,
        )
    }
}

#[rstest]
fn bit_width_must_fit_ptype(
    #[values(
        PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
        PType::I64
    )]
    ptype: PType,
    #[values(false, true)] too_wide: bool,
) -> VortexResult<()> {
    let bit_width = u8::try_from(ptype.bit_width())? + u8::from(too_wide);
    assert_eq!(single_block(ptype, bit_width, false).is_err(), too_wide);
    Ok(())
}

#[rstest]
fn block_offsets_can_fill_ptype(
    #[values(
        PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
        PType::I64
    )]
    ptype: PType,
) -> VortexResult<()> {
    single_block(ptype, u8::try_from(ptype.bit_width())?, true)?;
    Ok(())
}

#[cfg(debug_assertions)]
#[rstest]
#[should_panic(expected = "invalid BitPacked block offsets")]
fn too_wide_block_panics_in_debug(
    #[values(
        PType::U8, PType::I8, PType::U16, PType::I16, PType::U32, PType::I32, PType::U64,
        PType::I64
    )]
    ptype: PType,
) {
    let bit_width = u8::try_from(ptype.bit_width()).unwrap() + 1;
    single_block(ptype, bit_width, true).unwrap();
}

#[rstest]
#[case::equal_steps(buffer![0u64, 512, 1024])]
#[case::different_widths(buffer![0u64, 384, 1024])]
fn unsupported_offsets_leave_builder_unchanged(
    #[case] offsets: vortex_buffer::Buffer<u64>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(1024)),
        PType::U32,
        Validity::AllValid,
        None,
        offsets.into_array(),
        2048,
        0,
    )?;
    let mut builder = PrimitiveBuilder::<u32>::with_capacity_in(
        Nullability::Nullable,
        array.len() + 2,
        ctx.allocator(),
    );
    builder.append_null();
    assert!(array.append_to_builder(&mut builder, &mut ctx).is_err());
    builder.append_value(7);
    assert_arrays_eq!(
        builder.finish_into_primitive(),
        PrimitiveArray::from_option_iter([None, Some(7u32)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn construct_blocks_without_a_uniform_width() -> VortexResult<()> {
    // Two blocks at widths 3 and 4 occupy 896 bytes, which no constant width can represent.
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(896)),
        PType::U32,
        Validity::NonNullable,
        None,
        buffer![0u64, 384, 896].into_array(),
        2048,
        0,
    )?;
    assert!(matches!(array.bit_widths(), BitWidthsView::Blocked(_)));
    Ok(())
}

#[test]
fn empty_and_zero_width_arrays() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert!(matches!(
        encode(&[])?.bit_widths(),
        BitWidthsView::Global(0)
    ));
    let zeros = encode(&vec![0u32; 2049])?;
    assert!(matches!(zeros.bit_widths(), BitWidthsView::Global(0)));
    assert_eq!(zeros.packed().len(), 0);
    assert_arrays_eq!(zeros, PrimitiveArray::from_iter(vec![0u32; 2049]), &mut ctx);
    Ok(())
}

#[test]
fn into_parts_preserves_global_bit_width() -> VortexResult<()> {
    let parts = BitPacked::into_parts(uniform()?);
    assert!(matches!(parts.bit_widths, BitWidths::Global(7)));
    Ok(())
}

#[rstest]
#[case::equal_steps(buffer![128u64, 640, 1152].into_array())]
#[case::different_widths(buffer![128u64, 512, 1152].into_array())]
fn into_parts_preserves_block_offsets(#[case] offsets: ArrayRef) -> VortexResult<()> {
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::zeroed(1024)),
        PType::U32,
        Validity::NonNullable,
        None,
        offsets.clone(),
        1500,
        17,
    )?;
    let parts = BitPacked::into_parts(array);
    let BitWidths::Blocked(block_offsets) = parts.bit_widths else {
        vortex_bail!("expected block offsets");
    };
    assert!(ArrayRef::ptr_eq(&offsets, &block_offsets));
    let rebuilt = BitPacked::try_new_with_block_offsets(
        parts.packed,
        PType::U32,
        parts.validity,
        parts.patches,
        block_offsets,
        parts.len,
        parts.offset,
    )?;
    assert_eq!(rebuilt.len(), 1500);
    assert_eq!(rebuilt.offset(), 17);
    Ok(())
}
