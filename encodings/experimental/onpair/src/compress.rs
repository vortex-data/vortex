// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Offset;
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

use crate::OnPair;
use crate::OnPairArray;

/// Default OnPair training configuration: 12-bit codes ("dict-12").
pub const DEFAULT_DICT12_CONFIG: Config = onpair::DEFAULT_CONFIG;

/// Compress an iterable of optional byte strings via the OnPair encoder.
pub fn onpair_compress_iter<'a, I>(
    iter: I,
    len: usize,
    dtype: DType,
    config: Config,
) -> VortexResult<OnPairArray>
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    onpair_compress_iter_with_offsets::<u64, _>(iter, len, dtype, config)
}

fn onpair_compress_iter_with_offsets<'a, O, I>(
    iter: I,
    len: usize,
    dtype: DType,
    config: Config,
) -> VortexResult<OnPairArray>
where
    O: Offset,
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut flat: Vec<u8> = Vec::with_capacity(len * 16);
    let mut offsets: Vec<O> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    let mut validity_bits: Vec<bool> = Vec::with_capacity(len);
    offsets.push(<O as Offset>::from_usize(0));

    for item in iter {
        match item {
            Some(bytes) => {
                flat.extend_from_slice(bytes);
                offsets.push(<O as Offset>::from_usize(flat.len()));
                uncompressed_lengths.push(
                    i32::try_from(bytes.len()).vortex_expect("string length must fit in i32"),
                );
                validity_bits.push(true);
            }
            None => {
                offsets.push(<O as Offset>::from_usize(flat.len()));
                uncompressed_lengths.push(0);
                validity_bits.push(false);
            }
        }
    }

    let column = onpair::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let bits = column.bits;
    let dict_bytes = dict_bytes_to_buffer(column.dict_bytes);
    let codes_offsets = build_codes_offsets(&column.codes, &column.dict_offsets, &offsets)?;
    let codes = Buffer::from(column.codes).into_array();
    let dict_offsets = Buffer::from(column.dict_offsets).into_array();
    let codes_offsets = Buffer::from(codes_offsets).into_array();

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

/// Lift compressed dictionary bytes into the Vortex buffer slot.
fn dict_bytes_to_buffer(dict_bytes: Vec<u8>) -> BufferHandle {
    // Pad the dictionary blob with MAX_TOKEN_SIZE zero bytes so the
    // over-copy decoder can issue a fixed 16-byte load for every token
    // without risking an OOB read on the last entry.
    let mut padded = Vec::with_capacity(dict_bytes.len() + onpair::MAX_TOKEN_SIZE);
    padded.extend_from_slice(&dict_bytes);
    padded.resize(dict_bytes.len() + onpair::MAX_TOKEN_SIZE, 0);
    // Align dict_bytes to 8 bytes so the segment that ultimately holds the
    // OnPair tree starts at an 8-aligned in-memory address. Without this
    // anchor, the per-buffer padding the serializer inserts is only
    // *relative* to the segment start; if the segment lands at a u8-aligned
    // heap address, downstream `PrimitiveArray<u32>::deserialize` panics
    // with `Misaligned buffer cannot be used to build PrimitiveArray of u32`.
    BufferHandle::new_host(ByteBuffer::from(padded).aligned(vortex_buffer::Alignment::new(8)))
}

/// Reconstruct the per-row `codes_offsets` from the flat `codes`, the
/// dictionary `dict_offsets` (token byte lengths) and the per-row decoded byte
/// boundaries. Returns `nrows + 1` cumulative code counts (`u32`).
// TODO(joe): can we compute this while compressing the array, yes but a worse API.
fn build_codes_offsets<O: Offset>(
    codes: &[u16],
    dict_offsets: &[u32],
    row_byte_offsets: &[O],
) -> VortexResult<Vec<u32>> {
    let nrows = row_byte_offsets.len() - 1;
    let mut codes_offsets = Vec::with_capacity(nrows + 1);
    codes_offsets.push(0u32);
    let mut decoded_bytes: u64 = 0;
    let mut code_idx: usize = 0;
    for r in 0..nrows {
        let target = row_byte_offsets[r + 1]
            .to_usize()
            .ok_or_else(|| vortex_err!("OnPair row byte offset does not fit usize"))?
            as u64;
        while decoded_bytes < target {
            let code = codes[code_idx] as usize;
            decoded_bytes += u64::from(dict_offsets[code + 1] - dict_offsets[code]);
            code_idx += 1;
        }
        codes_offsets.push(
            u32::try_from(code_idx)
                .map_err(|_| vortex_err!("OnPair: code boundary {code_idx} does not fit u32"))?,
        );
    }
    Ok(codes_offsets)
}

/// Compress a byte-string accessor (typically a `VarBinArray` or
/// `VarBinViewArray`).
pub fn onpair_compress<A: ArrayAccessor<[u8]>>(
    array: A,
    len: usize,
    dtype: &DType,
    config: Config,
) -> VortexResult<OnPairArray> {
    array.with_iterator(|iter| onpair_compress_iter(iter, len, dtype.clone(), config))
}

/// Compress any [`ArrayRef`] whose canonical form is a string array, by first
/// canonicalising to `VarBinViewArray`.
pub fn onpair_compress_array(
    array: &ArrayRef,
    config: Config,
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
    config: Config,
) -> VortexResult<OnPairArray> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    onpair_compress_array(array, config, &mut ctx)
}
