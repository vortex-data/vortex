// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::FL_CHUNK_SIZE;

/// Transposes `bits` into FastLanes order, padding the result to whole 1,024-bit chunks.
///
/// Bit `i` of the result holds logical bit [`fastlanes::transpose(i)`](fastlanes::transpose), the
/// same element order [`fastlanes::Transpose::transpose`] gives a value vector. To address the
/// result by logical index instead, invert that with `crate::untranspose_idx`.
pub fn transpose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    bits_op(bits, fastlanes::transpose_bits::<u64>)
}

/// Untransposes whole 1,024-bit FastLanes chunks back into sequential bit order.
pub fn untranspose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    assert!(
        bits.len().is_multiple_of(FL_CHUNK_SIZE),
        "Transposed BitBuffer length must be a multiple of {FL_CHUNK_SIZE}"
    );
    bits_op(bits, fastlanes::untranspose_bits)
}

fn bits_op<F: Fn(&[u64; 16], &mut [u64; 16])>(bits: BitBuffer, op: F) -> BitBuffer {
    // Normalize to offset 0 with no bytes outside the logical range, so that byte-buffer chunk
    // boundaries line up with logical 1,024-bit chunks regardless of how the buffer was sliced.
    let bits = bits.sliced();
    let (_offset, len, bytes) = bits.into_inner();

    if len.is_multiple_of(FL_CHUNK_SIZE) && bytes.is_aligned(Alignment::of::<u64>()) {
        match bytes.try_into_mut() {
            Ok(mut bytes_mut) => {
                let (chunks, _) = bytes_mut.as_chunks_mut::<128>();
                let mut tmp = [0u64; 16];
                for chunk in chunks {
                    // SAFETY: the buffer is u64-aligned and chunks start at 128-byte multiples.
                    let chunk_u64 =
                        unsafe { mem::transmute::<&mut [u8; 128], &mut [u64; 16]>(chunk) };
                    op(chunk_u64, &mut tmp);
                    chunk_u64.copy_from_slice(&tmp);
                }
                BitBuffer::new(bytes_mut.freeze().into_byte_buffer(), len)
            }
            Err(bytes) => bits_op_with_copy(bytes, len, op),
        }
    } else {
        bits_op_with_copy(bytes, len, op)
    }
}

fn bits_op_with_copy<F: Fn(&[u64; 16], &mut [u64; 16])>(
    bytes: ByteBuffer,
    len: usize,
    op: F,
) -> BitBuffer {
    let output_len = bytes.len().div_ceil(8).next_multiple_of(16);
    let mut output = BufferMut::<u64>::with_capacity(output_len);
    let (input_chunks, input_trailer) = bytes.as_chunks::<128>();
    // Bound to the requested `output_len`: `spare_capacity_mut` may expose extra over-aligned
    // capacity, which would otherwise split into spurious trailing chunks and make `last_mut`
    // below target a chunk past the data we actually initialize.
    let (output_chunks, _) = unsafe {
        mem::transmute::<&mut [MaybeUninit<u64>], &mut [u64]>(
            &mut output.spare_capacity_mut()[..output_len],
        )
    }
    .as_chunks_mut::<16>();

    for (input, output) in input_chunks.iter().zip(output_chunks.iter_mut()) {
        op(&load_chunk_unaligned(input), output);
    }

    if !input_trailer.is_empty() {
        let mut padded_input = [0u8; 128];
        padded_input[0..input_trailer.len()].clone_from_slice(input_trailer);
        op(
            &load_chunk_unaligned(&padded_input),
            output_chunks
                .last_mut()
                .vortex_expect("Output wasn't a multiple of 128 bytes"),
        );
    }

    unsafe { output.set_len(output_len) };
    BitBuffer::new(
        output.freeze().into_byte_buffer(),
        len.next_multiple_of(FL_CHUNK_SIZE),
    )
}

/// Loads a 128-byte chunk into a `[u64; 16]` without requiring the input to be 8-byte aligned.
///
/// Native endianness is required: this must produce the same words as the aligned in-place path,
/// which reinterprets the buffer as host-endian `u64`s.
#[allow(clippy::host_endian_bytes)]
fn load_chunk_unaligned(chunk: &[u8; 128]) -> [u64; 16] {
    let mut words = [0u64; 16];
    let (bytes, _) = chunk.as_chunks::<8>();
    for (word, bytes) in words.iter_mut().zip(bytes) {
        *word = u64::from_ne_bytes(*bytes);
    }
    words
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;

    use super::*;

    fn make_validity_bits(num_bits: usize) -> BitBuffer {
        let mut builder = BitBufferMut::with_capacity(num_bits);
        for i in 0..num_bits {
            builder.append(i % 3 != 0);
        }
        builder.freeze()
    }

    fn force_copy_path(bits: BitBuffer) -> (BitBuffer, ByteBuffer) {
        let (offset, len, bytes) = bits.into_inner();
        let extra_ref = bytes.clone();
        (BitBuffer::new_with_offset(bytes, len, offset), extra_ref)
    }

    #[test]
    fn transpose_padding_copy_produces_same_bits() {
        let bits = make_validity_bits(500);
        let transposed = transpose_bitbuffer(bits.clone());
        assert_eq!(transposed.len(), 1024);
        let untransposed = untranspose_bitbuffer(transposed);
        assert_eq!(untransposed.slice(0..500), bits)
    }

    #[test]
    fn transpose_inplace_and_copy_produce_same_bits() {
        let bits = make_validity_bits(2048);

        let inplace_result = transpose_bitbuffer(bits.clone());

        let (bits_shared, _hold) = force_copy_path(bits);
        let copy_result = transpose_bitbuffer(bits_shared);

        assert_eq!(inplace_result.len(), copy_result.len());
        assert_eq!(inplace_result, copy_result);
    }

    #[test]
    fn transpose_bitbuffer_roundtrip_non_aligned() {
        let original_len = 1500;
        let bits = make_validity_bits(original_len);

        let transposed = transpose_bitbuffer(bits.clone());
        let roundtripped = untranspose_bitbuffer(transposed);
        assert_eq!(bits, roundtripped.slice(0..original_len));
    }

    /// Regression: the copy path split the over-aligned spare capacity into extra 128-byte
    /// chunks and wrote the padded remainder via `last_mut()`, which landed past the requested
    /// length and left the real trailing chunk uninitialized. Whether the surplus capacity
    /// produced an extra chunk depended on the allocation address, so we repeat across sizes to
    /// defeat that luck; the fix bounds the spare slice so the result no longer depends on it.
    #[test]
    fn transpose_copy_path_survives_overallocation() {
        for original_len in [129, 500, 1500, 9999] {
            let bits = make_validity_bits(original_len);
            for _ in 0..64 {
                let transposed = transpose_bitbuffer(bits.clone());
                let roundtripped = untranspose_bitbuffer(transposed);
                assert_eq!(
                    roundtripped.slice(0..original_len),
                    bits,
                    "len={original_len}"
                );
            }
        }
    }

    /// Regression: buffers that are not 8-byte aligned (e.g. views into file segments) must not
    /// panic and must produce the same result as the aligned path.
    #[test]
    fn untranspose_unaligned_buffer() {
        let bits = make_validity_bits(2048);
        let transposed = transpose_bitbuffer(bits.clone());
        let expected = untranspose_bitbuffer(transposed.clone());

        // Rebuild the transposed bytes at an odd offset within a larger allocation so the
        // resulting buffer is misaligned for u64.
        let (_, _, transposed_bytes) = transposed.sliced().into_inner();
        let mut shifted = ByteBufferMut::with_capacity(transposed_bytes.len() + 1);
        shifted.push(0xFF);
        shifted.extend_from_slice(&transposed_bytes);
        let misaligned = shifted.freeze().slice(1..transposed_bytes.len() + 1);
        assert!(!misaligned.is_aligned(Alignment::of::<u64>()));

        let untransposed = untranspose_bitbuffer(BitBuffer::new(misaligned, 2048));
        assert_eq!(untransposed, expected);
        assert_eq!(untransposed.slice(0..2048), bits);
    }

    /// Regression: a bit-offset view of transposed chunks (a sliced validity) must untranspose
    /// only the viewed chunks rather than assuming the buffer starts at a chunk boundary.
    #[test]
    fn untranspose_chunk_aligned_bit_offset() {
        let bits = make_validity_bits(3 * FL_CHUNK_SIZE);
        let transposed = transpose_bitbuffer(bits.clone());

        let view = transposed.slice(FL_CHUNK_SIZE..3 * FL_CHUNK_SIZE);
        let untransposed = untranspose_bitbuffer(view);
        assert_eq!(untransposed, bits.slice(FL_CHUNK_SIZE..3 * FL_CHUNK_SIZE));
    }
}
