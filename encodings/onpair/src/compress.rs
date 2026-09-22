// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::Config;
use onpair::Rows;
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
use vortex_mask::AllOr;

use crate::OnPair;
use crate::OnPairData;

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
    let mut uncompressed_lengths: BufferMut<u32> = BufferMut::zeroed(len);
    let buffers = array
        .data_buffers()
        .as_ref()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();

    // Keep sums local to each arm: a shared callback accumulator can prevent
    // vectorization of the all-valid loop.
    let total_bytes = match mask.bit_buffer() {
        AllOr::All => {
            let mut total = 0;
            for (view, length) in views.iter().zip(uncompressed_lengths.iter_mut()) {
                *length = view.len();
                total += *length as usize;
            }
            total
        }
        AllOr::None => unreachable!("all-null input handled above"),
        AllOr::Some(validity) => {
            let lengths = uncompressed_lengths.as_mut_slice();
            let mut total = 0;
            validity.for_each_set_index(|i| {
                let length = views[i].len();
                lengths[i] = length;
                total += length as usize;
            });
            total
        }
    };

    let rows = ViewRows {
        views,
        buffers: &buffers,
        lengths: uncompressed_lengths.as_slice(),
        total_bytes,
    };
    let column = onpair::compress_rows::<_, u64>(&rows, config);
    let (dict, codes, row_offsets) = column.into_raw();
    let (dict_bytes, dict_offsets) = dict.into_raw();
    let codes_offsets = codes_offsets_array(&row_offsets);
    let codes = Buffer::from(codes).into_array();
    // The `dict_offsets` child and the memoized widened-offsets cell share
    // this buffer, so seeding below costs no copy.
    let dict_offsets = Buffer::from(dict_offsets);

    let uncompressed_lengths = uncompressed_lengths.into_array();

    let data = OnPairData::try_new_with_dictionary(
        dict_bytes_to_buffer(dict_bytes),
        dict_offsets.clone(),
    )?;
    let encoded = OnPair::try_new_with_data(
        array.dtype().clone(),
        data,
        dict_offsets.into_array(),
        codes,
        codes_offsets,
        uncompressed_lengths,
        validity,
    )?;
    Ok(encoded.into_array())
}

/// Reads inline and external values in place, treating null rows as empty.
struct ViewRows<'a> {
    views: &'a [BinaryView],
    buffers: &'a [&'a ByteBuffer],
    lengths: &'a [u32],
    total_bytes: usize,
}

impl Rows for ViewRows<'_> {
    fn num_rows(&self) -> usize {
        self.views.len()
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    #[inline]
    fn row(&self, i: usize) -> &[u8] {
        let view = &self.views[i];
        if self.lengths[i] == 0 {
            &[]
        } else if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view_ref = view.as_view();
            &self.buffers[view_ref.buffer_index as usize][view_ref.as_range()]
        }
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
