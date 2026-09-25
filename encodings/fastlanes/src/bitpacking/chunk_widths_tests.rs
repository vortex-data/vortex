// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the chunk offsets child of uniformly bit-packed arrays.

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
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

#[test]
fn chunk_offsets_are_validated() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert_eq!(uniform.bit_width(), 7);
    assert_arrays_eq!(
        uniform.chunk_offsets(),
        vortex_buffer::buffer![0u64, 896, 1792, 2688].into_array(),
        &mut ctx
    );
    for offsets in [
        vortex_buffer::buffer![0u64, 896, 1792],
        vortex_buffer::buffer![0u64, 896, 1791, 2688],
        vortex_buffer::buffer![0u64, 896, 768, 2688],
        vortex_buffer::buffer![0u64, 768, 1792, 2688],
    ] {
        assert!(BitPacked::with_chunk_offsets(uniform.clone(), offsets.into_array()).is_err());
    }
    let wrong_dtype = PrimitiveArray::from_iter([0u32, 896, 1792, 2688]).into_array();
    assert!(BitPacked::with_chunk_offsets(uniform.clone(), wrong_dtype).is_err());
    let nullable =
        PrimitiveArray::from_option_iter([Some(0u64), Some(896), None, Some(2688)]).into_array();
    assert!(BitPacked::with_chunk_offsets(uniform.clone(), nullable).is_err());
    let rebased = vortex_buffer::buffer![128u64, 1024, 1920, 2816].into_array();
    let replaced = BitPacked::with_chunk_offsets(uniform.clone(), rebased)?;
    assert_arrays_eq!(uniform, replaced, &mut ctx);
    Ok(())
}

#[test]
fn empty_and_zero_width_offsets() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let empty = encode(&[])?;
    assert_arrays_eq!(
        empty.chunk_offsets(),
        vortex_buffer::buffer![0u64].into_array(),
        &mut ctx
    );
    let zeros = encode(&vec![0u32; 2049])?;
    assert_arrays_eq!(
        zeros.chunk_offsets(),
        vortex_buffer::buffer![0u64, 0, 0, 0].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(zeros, PrimitiveArray::from_iter(vec![0u32; 2049]), &mut ctx);
    Ok(())
}
