// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::NativePType;
use vortex_vector::binaryview::BinaryView;

/// Convert an offsets buffer to a buffer of element lengths.
#[inline]
pub fn offsets_to_lengths<P: NativePType>(offsets: &[P]) -> Buffer<P> {
    offsets
        .iter()
        .tuple_windows::<(_, _)>()
        .map(|(&start, &end)| end - start)
        .collect()
}

/// Maximum number of buffer bytes that can be referenced by a single `BinaryView`
pub const MAX_BUFFER_LEN: usize = i32::MAX as usize;

/// Split a large buffer of input `bytes` holding string data
pub fn build_views<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    mut bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());

    let mut buffers = Vec::new();
    let mut buf_index = start_buf_index;

    let mut offset = 0;
    for &len in lens {
        let len = len.as_();
        assert!(len <= max_buffer_len, "values cannot exceed max_buffer_len");

        if (offset + len) > max_buffer_len {
            // Roll the buffer every 2GiB, to avoid overflowing VarBinView offset field
            let rest = bytes.split_off(offset);

            buffers.push(bytes.freeze());
            buf_index += 1;
            offset = 0;

            bytes = rest;
        }
        let view = BinaryView::make_view(&bytes[offset..][..len], buf_index, offset.as_());
        // SAFETY: we reserved the right capacity beforehand
        unsafe { views.push_unchecked(view) };
        offset += len;
    }

    if !bytes.is_empty() {
        buffers.push(bytes.freeze());
    }

    (buffers, views.freeze())
}
