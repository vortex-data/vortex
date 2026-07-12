// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;

pub fn transpose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    let (offset, len, bytes) = bits.into_inner();
    let (bytes, output_len) = if bytes.len().is_multiple_of(128) {
        (bytes, len)
    } else {
        (pad_to_chunk(bytes), len.next_multiple_of(1024))
    };
    BitBuffer::new_with_offset(
        transform_chunks(bytes, fastlanes::transpose_bits),
        output_len,
        offset,
    )
}

pub fn untranspose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    assert!(
        bits.inner().len().is_multiple_of(128),
        "Transpose BitBuffer byte length must be a multiple of 128"
    );
    let (offset, len, bytes) = bits.into_inner();
    BitBuffer::new_with_offset(
        transform_chunks(bytes, fastlanes::untranspose_bits::<u64>),
        len,
        offset,
    )
}

fn transform_chunks(bytes: ByteBuffer, op: impl Fn(&[u64; 16], &mut [u64; 16])) -> ByteBuffer {
    let words = Buffer::<u64>::from_byte_buffer_aligned(
        bytes.aligned(Alignment::of::<u64>()),
        Alignment::of::<u64>(),
    );
    let mut words = words.into_mut();
    let (chunks, _) = words.as_chunks_mut::<16>();

    let mut output = [0u64; 16];
    for chunk in chunks {
        op(chunk, &mut output);
        chunk.copy_from_slice(&output);
    }
    words.freeze().into_byte_buffer()
}

fn pad_to_chunk(bytes: ByteBuffer) -> ByteBuffer {
    let mut padded = ByteBufferMut::zeroed(bytes.len().next_multiple_of(128));
    padded[..bytes.len()].copy_from_slice(&bytes);
    padded.freeze()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::ByteBuffer;

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
}
