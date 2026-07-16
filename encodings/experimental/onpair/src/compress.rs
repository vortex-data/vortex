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
    let codes_offsets = codes_offsets_array(&row_offsets, u32::MAX as usize);
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

/// Build the `codes_offsets` child from the library's per-row code boundaries,
/// storing the narrowest of `u32`/`u64` that holds the largest boundary.
/// `row_offsets` is non-decreasing, so its last entry is that maximum and one
/// bound check picks the width. `u32` covers the common case (the cascading
/// compressor narrows it further to `u16`/`u8`); `u64` engages only when a
/// single chunk carries more than `u32_max` tokens, matching the `u64` byte
/// offsets accepted at compression. `u32_max` is a parameter so tests can drive
/// the `u64` branch without a multi-GiB array.
fn codes_offsets_array<O: Offset>(row_offsets: &[O], u32_max: usize) -> ArrayRef {
    let total_tokens = row_offsets.last().map_or(0, |&o| o.to_usize());
    if total_tokens <= u32_max {
        Buffer::from(
            row_offsets
                .iter()
                .map(|&o| u32::try_from(o.to_usize()).vortex_expect("code boundary fits u32"))
                .collect::<Vec<u32>>(),
        )
        .into_array()
    } else {
        Buffer::from(
            row_offsets
                .iter()
                .map(|&o| u64::try_from(o.to_usize()).vortex_expect("token count fits u64"))
                .collect::<Vec<u64>>(),
        )
        .into_array()
    }
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

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use super::codes_offsets_array;

    #[test]
    fn codes_offsets_width_selection() {
        // Largest boundary within the threshold is stored as u32.
        let narrow = codes_offsets_array::<u64>(&[0, 3, 7], 7);
        assert_eq!(narrow.len(), 3);
        assert_eq!(
            narrow.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        // A boundary above the threshold widens the child to u64.
        let wide = codes_offsets_array::<u64>(&[0, 3, 8], 7);
        assert_eq!(
            wide.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable)
        );
    }
}
