// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{
    BitBuffer, BitBufferMut, BitView, get_bit, get_bit_unchecked, set_bit_unchecked,
    unset_bit_unchecked,
};
use vortex_mask::Mask;

use crate::filter::Filter;

impl Filter<Mask> for &BitBuffer {
    type Output = BitBuffer;

    fn filter(self, selection_mask: &Mask) -> BitBuffer {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match selection_mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => BitBuffer::empty(),
            Mask::Values(v) => {
                filter_indices(self.inner().as_ref(), self.offset(), v.indices()).freeze()
            }
        }
    }
}

impl Filter<Mask> for &mut BitBufferMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the mask length"
        );

        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(v) => {
                *self = filter_indices(self.inner().as_slice(), self.offset(), v.indices())
            }
        }
    }
}

fn filter_indices(bools: &[u8], bit_offset: usize, indices: &[usize]) -> BitBufferMut {
    // FIXME(ngates): this is slower than it could be!
    BitBufferMut::collect_bool(indices.len(), |idx| {
        let idx = *unsafe { indices.get_unchecked(idx) };
        get_bit(bools, bit_offset + idx)
    })
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &BitBuffer {
    type Output = BitBuffer;

    fn filter(self, selection: &BitView<'_, NB>) -> BitBuffer {
        let bits = self.inner().as_ptr();

        let mut out = BitBufferMut::with_capacity(selection.true_count());
        unsafe { out.set_len(selection.true_count()) };
        let mut out_idx = 0;
        selection.iter_ones(|idx| {
            let value = unsafe { get_bit_unchecked(bits, self.offset() + idx) };
            unsafe { out.set_to_unchecked(out_idx, value) };
            out_idx += 1;
        });
        out.freeze()
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &mut BitBufferMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_, NB>) {
        assert_eq!(
            self.len(),
            BitView::<NB>::N,
            "Selection mask length must equal the mask length"
        );

        let this = std::mem::take(self);

        let offset = this.offset();
        let mut buffer = this.into_inner();

        let buffer_ptr = buffer.as_mut_ptr();
        let mut out_idx = 0;
        selection.iter_ones(|idx| {
            let value = unsafe { get_bit_unchecked(buffer_ptr, offset + idx) };

            // NOTE(ngates): we don't call out.set_bit_unchecked here because it's nice that we
            //  can shift away any non-zero offset by writing directly into the bits buffer.
            if value {
                unsafe { set_bit_unchecked(buffer_ptr, out_idx) };
            } else {
                unsafe { unset_bit_unchecked(buffer_ptr, out_idx) };
            }
            out_idx += 1;
        });

        *self = BitBufferMut::from_buffer(buffer, 0, selection.true_count());
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::bitbuffer;

    use super::*;

    #[test]
    fn filter_bool_by_index_test() {
        let buf = bitbuffer![1 1 0];
        let filtered = filter_indices(buf.inner().as_ref(), 0, &[0, 2]).freeze();
        assert_eq!(2, filtered.len());
        assert_eq!(filtered, bitbuffer![1 0])
    }

    mod bitview_tests {
        use super::*;

        // Type aliases for commonly used BitView sizes
        type BitView8<'a> = BitView<'a, 8>; // 64 bits
        const N8: usize = BitView8::N; // 64 bits

        #[test]
        fn test_bitbuffer_filter_all_true() {
            let buf = BitBuffer::from_iter((0..N8).map(|i| i % 2 == 0));
            let view = BitView8::all_true();

            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), N8);
            assert_eq!(filtered, buf);
        }

        #[test]
        fn test_bitbuffer_filter_all_false() {
            let buf = BitBuffer::from_iter((0..N8).map(|i| i % 2 == 0));
            let view = BitView8::all_false();

            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), 0);
            assert_eq!(filtered, BitBuffer::empty());
        }

        #[test]
        fn test_bitbuffer_filter_prefix() {
            const NB: usize = 16; // 128 bits
            const N: usize = NB * 8;

            // Create a buffer with alternating bits
            let buf = BitBuffer::from_iter((0..N).map(|i| i % 2 == 0));

            // Filter to keep only first half
            let view = BitView::<NB>::with_prefix(N / 2);
            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), N / 2);
            let expected = BitBuffer::from_iter((0..N / 2).map(|i| i % 2 == 0));
            assert_eq!(filtered, expected);
        }

        #[test]
        fn test_bitbuffer_filter_sparse() {
            const NB: usize = 32; // 256 bits
            const N: usize = NB * 8;

            // Create a buffer with pattern 1010...
            let buf = BitBuffer::from_iter((0..N).map(|i| i % 2 == 0));

            // Select every 8th bit
            let mut bits = [0u8; NB];
            for i in (0..N).step_by(8) {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bits[byte_idx] |= 1 << bit_idx;
            }
            let view = BitView::<NB>::new(&bits);

            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), N / 8);
            // Every 8th position starting from 0: 0, 8, 16, 24, ...
            // Pattern at those positions: true, true, true, ...
            let expected = BitBuffer::from_iter((0..N).step_by(8).map(|i| i % 2 == 0));
            assert_eq!(filtered, expected);
        }

        #[test]
        fn test_bitbuffer_filter_alternating() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            // All true
            let buf = BitBuffer::new_set(N);

            // Alternating selection (01010101...)
            let mut bits = [0u8; NB];
            for i in 0..NB {
                bits[i] = 0b01010101;
            }
            let view = BitView::<NB>::new(&bits);

            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), N / 2);
            let expected = BitBuffer::new_set(N / 2);
            assert_eq!(filtered, expected);
        }

        #[test]
        fn test_bitbuffer_filter_custom_pattern() {
            const NB: usize = 16; // 128 bits
            const N: usize = NB * 8;

            // Pattern: 11001100...
            let buf = BitBuffer::from_iter((0..N).map(|i| (i / 2) % 2 == 0));

            // Select first 4 bits of each byte (00001111 pattern)
            let mut bits = [0u8; NB];
            for i in 0..NB {
                bits[i] = 0b00001111;
            }
            let view = BitView::<NB>::new(&bits);

            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), NB * 4); // 4 bits per byte
            // First 4 bits of each byte from pattern 11001100... = [1,1,0,0] repeated
            let expected =
                BitBuffer::from_iter((0..NB).flat_map(|_| [true, true, false, false].into_iter()));
            assert_eq!(filtered, expected);
        }

        #[test]
        fn test_bitbuffer_filter_with_offset() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            // Create buffer with offset
            let full_buf = BitBuffer::from_iter((0..N + 10).map(|i| i % 3 == 0));
            let buf = full_buf.slice(10..10 + N);

            let view = BitView::<NB>::with_prefix(32);
            let filtered = (&buf).filter(&view);

            assert_eq!(filtered.len(), 32);
            let expected = BitBuffer::from_iter((10..10 + 32).map(|i| i % 3 == 0));
            assert_eq!(filtered, expected);
        }

        #[test]
        fn test_bitbuffermut_filter_all_true() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::from_iter((0..N).map(|i| i % 2 == 0));
            let original = buf.clone().freeze();
            let view = BitView::<NB>::all_true();

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), N);
            assert_eq!(buf.freeze(), original);
        }

        #[test]
        fn test_bitbuffermut_filter_all_false() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::from_iter((0..N).map(|i| i % 2 == 0));
            let view = BitView::<NB>::all_false();

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), 0);
        }

        #[test]
        fn test_bitbuffermut_filter_prefix() {
            const NB: usize = 16; // 128 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::from_iter((0..N).map(|i| i % 2 == 0));
            let view = BitView::<NB>::with_prefix(N / 2);

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), N / 2);
            let expected = BitBufferMut::from_iter((0..N / 2).map(|i| i % 2 == 0));
            assert_eq!(buf.freeze(), expected.freeze());
        }

        #[test]
        fn test_bitbuffermut_filter_sparse() {
            const NB: usize = 32; // 256 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::from_iter((0..N).map(|i| i % 2 == 0));

            // Select every 8th bit
            let mut bits = [0u8; NB];
            for i in (0..N).step_by(8) {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bits[byte_idx] |= 1 << bit_idx;
            }
            let view = BitView::<NB>::new(&bits);

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), N / 8);
            let expected = BitBufferMut::from_iter((0..N).step_by(8).map(|i| i % 2 == 0));
            assert_eq!(buf.freeze(), expected.freeze());
        }

        #[test]
        fn test_bitbuffermut_filter_alternating() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::new_set(N);

            // Alternating selection (01010101...)
            let mut bits = [0u8; NB];
            for i in 0..NB {
                bits[i] = 0b01010101;
            }
            let view = BitView::<NB>::new(&bits);

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), N / 2);
            assert_eq!(buf.freeze(), BitBuffer::new_set(N / 2));
        }

        #[test]
        fn test_bitbuffermut_filter_inverse_alternating() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            let mut buf = BitBufferMut::new_set(N);

            // Inverse alternating selection (10101010...)
            let mut bits = [0u8; NB];
            for i in 0..NB {
                bits[i] = 0b10101010;
            }
            let view = BitView::<NB>::new(&bits);

            (&mut buf).filter(&view);

            assert_eq!(buf.len(), N / 2);
            assert_eq!(buf.freeze(), BitBuffer::new_set(N / 2));
        }

        #[test]
        fn test_bitbuffermut_filter_complex_pattern() {
            const NB: usize = 16; // 128 bits
            const N: usize = NB * 8;

            // Create pattern: every 3rd bit is true
            let mut buf = BitBufferMut::from_iter((0..N).map(|i| i % 3 == 0));

            // Select using prime stepping (every 7th bit)
            let mut bits = [0u8; NB];
            let mut idx = 0;
            while idx < N {
                let byte_idx = idx / 8;
                let bit_idx = idx % 8;
                bits[byte_idx] |= 1 << bit_idx;
                idx += 7;
            }
            let view = BitView::<NB>::new(&bits);

            (&mut buf).filter(&view);

            // Verify we got the right length
            let expected_len = (0..N).step_by(7).count();
            assert_eq!(buf.len(), expected_len);

            // Verify the values are correct
            let expected = BitBufferMut::from_iter((0..N).step_by(7).map(|i| i % 3 == 0));
            assert_eq!(buf.freeze(), expected.freeze());
        }

        #[test]
        fn test_bitbuffermut_filter_with_offset() {
            const NB: usize = 8; // 64 bits
            const N: usize = NB * 8;

            // Create buffer with bit offset
            let full_buf = BitBufferMut::from_iter((0..N + 10).map(|i| i % 3 == 0));
            let frozen = full_buf.freeze();
            let mut buf = frozen.slice(10..10 + N).into_mut();

            let view = BitView::<NB>::with_prefix(32);
            (&mut buf).filter(&view);

            assert_eq!(buf.len(), 32);
            let expected = BitBufferMut::from_iter((10..10 + 32).map(|i| i % 3 == 0));
            assert_eq!(buf.freeze(), expected.freeze());
        }

        #[test]
        fn test_different_nb_sizes() {
            // Test with NB = 8 (64 bits)
            {
                const NB: usize = 8;
                const N: usize = NB * 8;
                let buf = BitBuffer::new_set(N);
                let view = BitView::<NB>::with_prefix(N / 2);
                let filtered = (&buf).filter(&view);
                assert_eq!(filtered.len(), N / 2);
            }

            // Test with NB = 64 (512 bits)
            {
                const NB: usize = 64;
                const N: usize = NB * 8;
                let buf = BitBuffer::new_set(N);
                let view = BitView::<NB>::with_prefix(N / 2);
                let filtered = (&buf).filter(&view);
                assert_eq!(filtered.len(), N / 2);
            }

            // Test with NB = 128 (1024 bits)
            {
                const NB: usize = 128;
                const N: usize = NB * 8;
                let buf = BitBuffer::new_set(N);
                let view = BitView::<NB>::with_prefix(N / 2);
                let filtered = (&buf).filter(&view);
                assert_eq!(filtered.len(), N / 2);
            }
        }
    }
}
