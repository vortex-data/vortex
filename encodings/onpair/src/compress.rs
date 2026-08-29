// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train + compress entry points for the OnPair encoding.

use onpair::CompactDictionary;
use onpair::Config;
use onpair::Offset;
use onpair::Rows;
use onpair::Token;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
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
    // `onpair::compress_rows` uses one offset width for the per-row code offsets it returns, and a
    // row emits at most one code per input byte. Asking for `u32` when the column's bytes fit
    // it therefore halves that buffer and lets `codes_offsets` adopt the returned vector
    // instead of narrowing it in a second pass. `onpair` narrows offsets with a plain `as`
    // cast, so this byte-count check is what keeps the `u32` path from truncating.
    let (dict, codes, codes_offsets) = if u32::try_from(total_bytes).is_ok() {
        compress_column::<u32>(&rows, config)
    } else {
        compress_column::<u64>(&rows, config)
    };
    let (dict_bytes, dict_offsets) = dict.into_raw();
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

/// Compress `rows` at the given code-offset width, returning the dictionary, the code
/// stream, and the lowered `codes_offsets` child.
fn compress_column<O: CodeOffset>(
    rows: &ViewRows<'_>,
    config: Config,
) -> (CompactDictionary, Vec<Token>, ArrayRef) {
    let column = onpair::compress_rows::<_, O>(rows, config);
    let (dict, codes, row_offsets) = column.into_raw();
    (dict, codes, O::codes_offsets_array(row_offsets))
}

/// The code-offset widths [`onpair::compress_rows`] can return, plus how each one lowers
/// those offsets into the `codes_offsets` child.
trait CodeOffset: Offset + NativePType {
    /// Build the `codes_offsets` child from the per-row code boundaries.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef
    where
        Self: Sized;
}

impl CodeOffset for u32 {
    /// Already the narrowest width Vortex stores, so adopt the vector as-is. The cascading
    /// compressor narrows it further to `u16`/`u8`.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef {
        Buffer::from(row_offsets).into_array()
    }
}

impl CodeOffset for u64 {
    /// Reached only when a column carries more than `u32::MAX` bytes. Tokens are still often
    /// far fewer, so keep the narrowing pass rather than storing `u64` offsets that do not
    /// need the range. `row_offsets` is non-decreasing, so its last entry is the maximum and
    /// one bound check picks the width.
    fn codes_offsets_array(row_offsets: Vec<Self>) -> ArrayRef {
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
            Buffer::from(row_offsets).into_array()
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::*;
    use crate::array::OnPairArraySlotsExt;

    /// A column under `u32::MAX` bytes must keep storing `codes_offsets` at `u32`. This guards
    /// the stored width across the switch to requesting `u32` from `onpair` directly; whether a
    /// narrowing pass ran is not observable from here.
    #[test]
    fn codes_offsets_are_u32_for_small_inputs() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str([
            "the quick brown fox",
            "jumps over the lazy dog",
            "the quick brown fox jumps",
        ])
        .into_array();

        let encoded = onpair_compress(&array, Config::default(), &mut ctx)?;
        let onpair = encoded
            .as_opt::<OnPair>()
            .vortex_expect("input compresses to OnPair");
        assert_eq!(onpair.codes_offsets().dtype().as_ptype(), PType::U32);
        Ok(())
    }

    /// `u64` code offsets still narrow to `u32` when the token count allows it, so the
    /// wide-input path stores the same width it did before.
    #[test]
    fn u64_code_offsets_narrow_to_u32() {
        let array = <u64 as CodeOffset>::codes_offsets_array(vec![0, 3, 7, 11]);
        assert_eq!(array.dtype().as_ptype(), PType::U32);
    }
}
