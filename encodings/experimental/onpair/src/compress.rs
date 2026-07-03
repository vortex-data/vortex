// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Offset;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;

use crate::OnPair;
use crate::OnPairArray;

/// Default OnPair training configuration: 12-bit codes ("dict-12").
pub const DEFAULT_DICT12_CONFIG: Config = onpair::DEFAULT_CONFIG;

fn onpair_compress_varbinview<O>(
    array: VarBinViewArray,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray>
where
    O: Offset,
{
    let len = array.len();
    let mask = array.validity()?.execute_mask(len, ctx)?;
    if mask.all_false() {
        return OnPair::try_new(
            array.dtype().clone(),
            BufferHandle::new_host(ByteBuffer::empty()),
            ConstantArray::new(0, len).into_array(),
            ConstantArray::new(0u16, len).into_array(),
            ConstantArray::new(0u32, len + 1).into_array(),
            ConstantArray::new(0i32, len).into_array(),
            Validity::AllInvalid,
            9,
        );
    }

    let mut flat: Vec<u8> = Vec::with_capacity(len * 16);
    let mut offsets: Vec<O> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    offsets.push(O::from_usize(0));
    let views = array.views();
    let buffers = array
        .data_buffers()
        .as_ref()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();

    match mask.bit_buffer() {
        AllOr::All => {
            for view in views {
                let bytes = view_bytes(view, &buffers);
                flat.extend_from_slice(bytes);
                offsets.push(O::from_usize(flat.len()));
                uncompressed_lengths
                    .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
            }
        }
        AllOr::None => {
            unreachable!("all_false() should have been caught earlier");
        }
        AllOr::Some(validity) => {
            for (view, valid) in views.iter().zip(validity.iter()) {
                if valid {
                    let bytes = view_bytes(view, &buffers);
                    flat.extend_from_slice(bytes);
                    offsets.push(O::from_usize(flat.len()));
                    uncompressed_lengths
                        .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                } else {
                    offsets.push(O::from_usize(flat.len()));
                    uncompressed_lengths.push(0);
                }
            }
        }
    }

    let column = onpair::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let bits = column.bits;
    let dict_bytes = dict_bytes_to_buffer(column.dict_bytes);
    let codes_offsets =
        build_codes_offsets(&column.codes, &column.dict_offsets, &offsets)?.into_array();
    let codes = Buffer::from(column.codes).into_array();
    let dict_offsets = Buffer::from(column.dict_offsets).into_array();

    let uncompressed_lengths = uncompressed_lengths.into_array();

    OnPair::try_new(
        array.dtype().clone(),
        dict_bytes,
        dict_offsets,
        codes,
        codes_offsets,
        uncompressed_lengths,
        array.validity()?,
        bits,
    )
}

fn view_bytes<'a>(view: &'a BinaryView, buffers: &'a [&ByteBuffer]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view_ref = view.as_view();
        &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
    }
}

/// Lift compressed dictionary bytes into the Vortex buffer slot.
fn dict_bytes_to_buffer(dict_bytes: Vec<u8>) -> BufferHandle {
    // Pad the dictionary blob with MAX_TOKEN_SIZE zero bytes so the
    // over-copy decoder can issue a fixed 16-byte load for every token
    // without risking an OOB read on the last entry.
    //
    // Align dict_bytes to 8 bytes so the segment that ultimately holds the
    // OnPair tree starts at an 8-aligned in-memory address. Without this
    // anchor, the per-buffer padding the serializer inserts is only
    // *relative* to the segment start; if the segment lands at a u8-aligned
    // heap address, downstream `PrimitiveArray<u32>::deserialize` panics
    // with `Misaligned buffer cannot be used to build PrimitiveArray of u32`.
    let mut padded = ByteBufferMut::with_capacity_aligned(
        dict_bytes.len() + onpair::MAX_TOKEN_SIZE,
        Alignment::new(8),
    );
    padded.extend_from_slice(&dict_bytes);
    unsafe { padded.push_n_unchecked(0, dict_bytes.len() + onpair::MAX_TOKEN_SIZE - padded.len()) };
    BufferHandle::new_host(padded.freeze())
}

/// Reconstruct the per-row `codes_offsets` from the flat `codes`, the
/// dictionary `dict_offsets` (token byte lengths) and the per-row decoded byte
/// boundaries. Returns `nrows + 1` cumulative code counts (`u32`).
// TODO(joe): can we compute this while compressing the array, yes but a worse API.
fn build_codes_offsets<O: Offset>(
    codes: &[u16],
    dict_offsets: &[u32],
    row_byte_offsets: &[O],
) -> VortexResult<Buffer<u32>> {
    let nrows = row_byte_offsets.len() - 1;
    let mut codes_offsets = BufferMut::with_capacity(nrows + 1);
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
    Ok(codes_offsets.freeze())
}

/// Compress any [`ArrayRef`] whose canonical form is a string array, by first
/// canonicalising to `VarBinViewArray`.
pub fn onpair_compress(
    array: &ArrayRef,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray> {
    let view = array.clone().execute::<VarBinViewArray>(ctx)?;
    onpair_compress_varbinview::<u64>(view, config, ctx)
}
