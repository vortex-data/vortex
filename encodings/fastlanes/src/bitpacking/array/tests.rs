// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for construction and validation of [`BitPacked`] arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
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

fn single_block(ptype: PType, bit_width: u8) -> VortexResult<BitPackedArray> {
    let packed = BufferHandle::new_host(ByteBuffer::zeroed(128 * usize::from(bit_width)));
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
    assert_eq!(single_block(ptype, bit_width).is_err(), too_wide);
    Ok(())
}

#[test]
fn empty_and_zero_width_arrays() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(encode(&[])?.bit_width(), 0);
    let zeros = encode(&vec![0u32; 2049])?;
    assert_eq!(zeros.bit_width(), 0);
    assert_eq!(zeros.packed().len(), 0);
    assert_arrays_eq!(zeros, PrimitiveArray::from_iter(vec![0u32; 2049]), &mut ctx);
    Ok(())
}

#[test]
fn into_parts_preserves_bit_width() -> VortexResult<()> {
    let parts = BitPacked::into_parts(uniform()?);
    assert_eq!(parts.bit_width, 7);
    Ok(())
}

#[test]
fn test_all_null_slice_keeps_all_invalid_validity() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let bitpacked = BitPacked::try_new(
        BufferHandle::new_host(ByteBuffer::zeroed(1024)),
        PType::U32,
        Validity::AllInvalid,
        None,
        8,
        1024,
        0,
    )?;
    let decoded = bitpacked
        .slice(10..20)?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert!(matches!(decoded.validity()?, Validity::AllInvalid));
    Ok(())
}
