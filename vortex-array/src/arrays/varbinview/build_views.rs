// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;

pub use crate::arrays::BinaryView;
use crate::dtype::NativePType;

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

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;

    use crate::arrays::BinaryView;
    use crate::arrays::build_views::build_views;

    #[test]
    fn test_to_canonical_large() {
        // We are testing generating views for raw data that should look like
        //
        //    aaaaaaaaaaaaa ("a"*13)
        //    bbbbbbbbbbbbb ("b"*13)
        //    ccccccccccccc ("c"*13)
        //    ddddddddddddd ("d"*13)
        //
        // In real code, this would all fit in one buffer, but to unit test the splitting logic
        // we split buffers at length 26, which should result in two buffers for the output array.
        let raw_data =
            ByteBufferMut::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbbcccccccccccccddddddddddddd");
        let lens = vec![13u8; 4];

        let (buffers, views) = build_views(0, 26, raw_data, &lens);

        assert_eq!(
            buffers,
            vec![
                ByteBuffer::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbb"),
                ByteBuffer::copy_from("cccccccccccccddddddddddddd"),
            ]
        );

        assert_eq!(
            views.as_slice(),
            &[
                BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 0),
                BinaryView::make_view(b"bbbbbbbbbbbbb", 0, 13),
                BinaryView::make_view(b"ccccccccccccc", 1, 0),
                BinaryView::make_view(b"ddddddddddddd", 1, 13),
            ]
        )
    }
}
