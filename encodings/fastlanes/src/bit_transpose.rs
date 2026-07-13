// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

pub fn transpose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    let (offset, len, bytes) = bits.into_inner();
    BitBuffer::new_with_offset(
        transform_buffer(bytes, fastlanes::transpose_bits),
        len.next_multiple_of(1024),
        offset,
    )
}

pub fn untranspose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    assert!(
        bits.inner().len().is_multiple_of(128),
        "Transpose BitBuffer byte length must be a multiple of 128"
    );
    assert!(
        bits.inner().is_aligned(Alignment::of::<u64>()),
        "Transposed BitBuffer must be 8 byte aligned"
    );
    let (offset, len, bytes) = bits.into_inner();
    BitBuffer::new_with_offset(
        transform_buffer(bytes, fastlanes::untranspose_bits::<u64>),
        len,
        offset,
    )
}

fn transform_buffer(bytes: ByteBuffer, op: impl Fn(&[u64; 16], &mut [u64; 16])) -> ByteBuffer {
    let alignment = Alignment::of::<u64>();

    if bytes.is_aligned(alignment) {
        let words = Buffer::<u64>::from_byte_buffer_aligned(bytes, alignment);
        match words.try_into_mut() {
            Ok(words_mut) => transform_in_place(words_mut, op),
            Err(words) => transform_copy(words.into_byte_buffer(), op),
        }
    } else {
        transform_copy(bytes, op)
    }
}

fn transform_copy(bytes: ByteBuffer, op: impl Fn(&[u64; 16], &mut [u64; 16])) -> ByteBuffer {
    let out_len = bytes.len().next_multiple_of(128);
    let mut words_out = BufferMut::<u64>::with_capacity(out_len);
    let (in_chunks, trailer) = bytes.as_chunks::<128>();
    let (out_chunks, _) = words_out.spare_capacity_mut().as_chunks_mut::<16>();

    for (chunk, output) in in_chunks
        .iter()
        .zip(out_chunks[..in_chunks.len() - 1].iter_mut())
    {
        op(
            unsafe { mem::transmute::<&[u8; 128], &[u64; 16]>(chunk) },
            unsafe { mem::transmute::<&mut [MaybeUninit<u64>; 16], &mut [u64; 16]>(output) },
        );
    }

    if !trailer.is_empty() {
        let mut padded_input = [0u8; 128];
        padded_input[0..trailer.len()].clone_from_slice(trailer);
        op(
            unsafe { mem::transmute::<&[u8; 128], &[u64; 16]>(&padded_input) },
            unsafe {
                mem::transmute::<&mut [MaybeUninit<u64>; 16], &mut [u64; 16]>(
                    out_chunks
                        .last_mut()
                        .vortex_expect("Output wasn't a multiple of 128 bytes"),
                )
            },
        );
    }

    unsafe { words_out.set_len(out_len) };
    words_out.freeze().into_byte_buffer()
}

fn transform_in_place(
    mut words: BufferMut<u64>,
    op: impl Fn(&[u64; 16], &mut [u64; 16]),
) -> ByteBuffer {
    let (chunks, _) = words.as_chunks_mut::<16>();

    let mut output = [0u64; 16];
    for chunk in chunks {
        op(chunk, &mut output);
        chunk.copy_from_slice(&output);
    }
    words.freeze().into_byte_buffer()
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
