// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::scalar::Scalar;
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

/// Default OnPair training configuration: 12-bit codes ("dict-12").
pub const DEFAULT_DICT12_CONFIG: Config = Config {
    seed: Some(42),
    ..onpair::DEFAULT_CONFIG
};

/// Compress any [`ArrayRef`] whose canonical form is a string array.
///
/// All-null inputs are returned as a [`ConstantArray`].
pub fn onpair_compress(
    array: &ArrayRef,
    config: Config,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let len = array.len();
    let validity = array.validity()?;
    let mask = validity.execute_mask(len, ctx)?;
    if matches!(mask.bit_buffer(), AllOr::None) {
        // CascadingCompressor handles this earlier, but direct callers can reach it.
        return Ok(ConstantArray::new(Scalar::null(array.dtype().clone()), len).into_array());
    }

    let views = array.views();
    let flat_bytes: usize = views.iter().map(|v| v.len() as usize).sum();

    // TODO(francesco): we flatten because onpair training needs a contiguous `(bytes, offsets)`
    // pair. Allowing onpair to train on a slice-of-slices would let us skip this copy.
    let mut flat: Vec<u8> = Vec::with_capacity(flat_bytes);
    let mut offsets: Vec<u64> = Vec::with_capacity(len + 1);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    offsets.push(0);
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
                offsets.push(u64::try_from(flat.len()).vortex_expect("offset must fit in u64"));
                uncompressed_lengths
                    .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
            }
        }
        AllOr::None => unreachable!("all-null input handled above"),
        AllOr::Some(validity) => {
            for (view, valid) in views.iter().zip(validity.iter()) {
                if valid {
                    let bytes = view_bytes(view, &buffers);
                    flat.extend_from_slice(bytes);
                    offsets.push(u64::try_from(flat.len()).vortex_expect("offset must fit in u64"));
                    uncompressed_lengths
                        .push(i32::try_from(view.len()).vortex_expect("must fit in i32"));
                } else {
                    offsets.push(u64::try_from(flat.len()).vortex_expect("offset must fit in u64"));
                    uncompressed_lengths.push(0);
                }
            }
        }
    }

    let column = onpair::compress(&flat, &offsets, config)
        .map_err(|e| vortex_err!("OnPair compress failed: {e}"))?;
    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    let codes_offsets = codes_offsets_array(&row_offsets);
    let codes = Buffer::from(codes).into_array();
    // The `dict_offsets` child and the memoized widened-offsets cell share
    // this buffer, so seeding below costs no copy.
    let dict_offsets = Buffer::from(dict_offsets);

    let uncompressed_lengths = uncompressed_lengths.into_array();

    let encoded = OnPair::try_new(
        array.dtype().clone(),
        dict_bytes_to_buffer(dict_bytes),
        dict_offsets.clone().into_array(),
        codes,
        codes_offsets,
        uncompressed_lengths,
        validity,
    )?;
    // SAFETY: the trainer's dictionary is conformant by construction, and
    // `dict_offsets` is exactly the u32 child attached above.
    unsafe { encoded.init_dict_offsets(dict_offsets) };
    Ok(encoded.into_array())
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

/// Build the `codes_offsets` child from the library's per-row code boundaries,
/// storing the narrowest of `u32`/`u64` that holds the largest boundary.
/// `row_offsets` is non-decreasing, so its last entry is that maximum and one
/// bound check picks the width. `u32` covers the common case (the cascading
/// compressor narrows it further to `u16`/`u8`); `u64` engages only when a
/// single chunk carries more than `u32::MAX` tokens, matching the `u64` byte
/// offsets accepted at compression.
fn codes_offsets_array(row_offsets: &[u64]) -> ArrayRef {
    let total_tokens = row_offsets.last().copied().unwrap_or(0);
    if u32::try_from(total_tokens).is_ok() {
        Buffer::from(
            row_offsets
                .iter()
                .map(|&o| u32::try_from(o).vortex_expect("code boundary fits u32"))
                .collect::<Vec<u32>>(),
        )
        .into_array()
    } else {
        Buffer::from(row_offsets.to_vec()).into_array()
    }
}
