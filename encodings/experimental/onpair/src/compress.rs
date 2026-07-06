// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Offset;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
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
pub const DEFAULT_DICT12_CONFIG: Config = Config {
    seed: Some(42),
    ..onpair::DEFAULT_CONFIG
};

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
            offsets.resize(len + 1, O::from_usize(0));
            for _ in 0..len {
                uncompressed_lengths.push(0);
            }
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
    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    let codes_offsets = Buffer::from(
        row_offsets
            .into_iter()
            .map(|o| {
                let value = o.to_usize();
                u32::try_from(value)
                    .map_err(|_| vortex_err!("OnPair code boundary {value} does not fit u32"))
            })
            .collect::<VortexResult<Vec<_>>>()?,
    )
    .into_array();
    let codes = Buffer::from(codes).into_array();
    let dict_offsets = Buffer::from(dict_offsets).into_array();

    let uncompressed_lengths = uncompressed_lengths.into_array();

    OnPair::try_new(
        array.dtype().clone(),
        dict_bytes_to_buffer(dict_bytes),
        dict_offsets,
        codes,
        codes_offsets,
        uncompressed_lengths,
        array.validity()?,
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

fn dict_bytes_to_buffer(dict_bytes: Vec<u8>) -> BufferHandle {
    // Align dict_bytes to 8 bytes so the segment that ultimately holds the
    // OnPair tree starts at an 8-aligned in-memory address. Without this anchor,
    // downstream primitive children may deserialize from a misaligned segment.
    let mut aligned = ByteBufferMut::with_capacity_aligned(dict_bytes.len(), Alignment::new(8));
    aligned.extend_from_slice(&dict_bytes);
    BufferHandle::new_host(aligned.freeze())
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
