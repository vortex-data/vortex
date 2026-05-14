// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_onpair_sys::Column;
use vortex_onpair_sys::OnPairTrainingConfig;
use vortex_onpair_sys::unpack_codes_to_u16;

use crate::OnPair;
use crate::OnPairArray;

/// Default OnPair training configuration: 12-bit codes ("dict-12").
pub const DEFAULT_DICT12_CONFIG: OnPairTrainingConfig = vortex_onpair_sys::DEFAULT_DICT12_CONFIG;

/// Build a training config with a custom bit width.
pub fn config_with_bits(bits: u32) -> OnPairTrainingConfig {
    OnPairTrainingConfig {
        bits,
        threshold: 0.5,
        seed: 0,
    }
}

/// Compress an iterable of optional byte strings via the OnPair C++ library.
pub fn onpair_compress_iter<'a, I>(
    iter: I,
    len: usize,
    dtype: DType,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray>
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut flat: Vec<u8> = Vec::with_capacity(len * 16);
    let mut offsets: Vec<u64> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    let mut validity_bits: Vec<bool> = Vec::with_capacity(len);
    offsets.push(0);

    for item in iter {
        match item {
            Some(bytes) => {
                flat.extend_from_slice(bytes);
                offsets.push(flat.len() as u64);
                uncompressed_lengths.push(
                    i32::try_from(bytes.len()).vortex_expect("string length must fit in i32"),
                );
                validity_bits.push(true);
            }
            None => {
                offsets.push(flat.len() as u64);
                uncompressed_lengths.push(0);
                validity_bits.push(false);
            }
        }
    }

    let column = Column::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let (bits, dict_bytes, dict_offsets, codes, codes_offsets) = parts_to_children(&column)?;
    drop(column);

    let uncompressed_lengths = uncompressed_lengths.into_array();
    let validity = match dtype.nullability() {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::from_iter(validity_bits),
    };

    OnPair::try_new(
        dtype,
        dict_bytes,
        dict_offsets,
        codes,
        codes_offsets,
        uncompressed_lengths,
        validity,
        bits,
    )
}

/// Borrow the raw C++ parts and lift them into Vortex children + the dict buffer.
/// Returns `(bits, dict_bytes_buffer, dict_offsets_child, codes_child, codes_offsets_child)`.
fn parts_to_children(
    column: &Column,
) -> VortexResult<(u32, BufferHandle, ArrayRef, ArrayRef, ArrayRef)> {
    let parts = column
        .parts()
        .map_err(|e| vortex_err!("OnPair parts failed: {e}"))?;
    let bits = parts.bits;
    // Pad the dictionary blob with MAX_TOKEN_SIZE zero bytes so the
    // over-copy decoder can issue a fixed 16-byte load for every token
    // without risking an OOB read on the last entry.
    let mut padded = Vec::with_capacity(parts.dict_bytes.len() + crate::MAX_TOKEN_SIZE);
    padded.extend_from_slice(parts.dict_bytes);
    padded.resize(parts.dict_bytes.len() + crate::MAX_TOKEN_SIZE, 0);
    // Align dict_bytes to 8 bytes so the segment that ultimately holds the
    // OnPair tree starts at an 8-aligned in-memory address. Without this
    // anchor, the per-buffer padding the serializer inserts is only
    // *relative* to the segment start; if the segment lands at a u8-aligned
    // heap address, downstream `PrimitiveArray<u32>::deserialize` panics
    // with `Misaligned buffer cannot be used to build PrimitiveArray of u32`.
    let dict_bytes =
        BufferHandle::new_host(ByteBuffer::from(padded).aligned(vortex_buffer::Alignment::new(8)));

    let dict_offsets = Buffer::<u32>::copy_from(parts.dict_offsets).into_array();
    let total_tokens = usize::try_from(
        *parts
            .codes_boundaries
            .last()
            .ok_or_else(|| vortex_err!("OnPair: missing codes_boundaries"))?,
    )
    .map_err(|_| vortex_err!("OnPair: total_tokens does not fit in usize"))?;
    let codes_vec = unpack_codes_to_u16(parts.codes_packed, total_tokens, bits);
    let codes = Buffer::<u16>::copy_from(codes_vec).into_array();
    let codes_offsets = Buffer::<u32>::copy_from(parts.codes_boundaries).into_array();
    Ok((bits, dict_bytes, dict_offsets, codes, codes_offsets))
}

/// Compress a byte-string accessor (typically a `VarBinArray` or
/// `VarBinViewArray`).
pub fn onpair_compress<A: ArrayAccessor<[u8]>>(
    array: A,
    len: usize,
    dtype: &DType,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray> {
    array.with_iterator(|iter| onpair_compress_iter(iter, len, dtype.clone(), config))
}

/// Compress any [`ArrayRef`] whose canonical form is a string array, by first
/// canonicalising to `VarBinViewArray`.
pub fn onpair_compress_array(
    array: &ArrayRef,
    config: OnPairTrainingConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray> {
    let view = array.clone().execute::<VarBinViewArray>(ctx)?;
    let len = view.len();
    let dtype = view.dtype().clone();
    onpair_compress(&view, len, &dtype, config)
}

/// Convenience: build a default `ExecutionCtx` from `LEGACY_SESSION`.
pub fn onpair_compress_array_default(
    array: &ArrayRef,
    config: OnPairTrainingConfig,
) -> VortexResult<OnPairArray> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    onpair_compress_array(array, config, &mut ctx)
}
