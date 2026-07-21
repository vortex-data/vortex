// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

pub fn transpose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    let (offset, len, bytes) = bits.into_inner();

    if bytes.len().is_multiple_of(128) {
        match bytes.try_into_mut() {
            Ok(mut bytes_mut) => {
                // We can ignore the spare trailer capacity that can be an artifact of allocator as we requested 128 multiple chunks
                let (chunks, _) = bytes_mut.as_chunks_mut::<128>();
                let mut tmp = [0u64; 16];
                for chunk in chunks {
                    let chunk_u64 =
                        unsafe { mem::transmute::<&mut [u8; 128], &mut [u64; 16]>(chunk) };
                    fastlanes::transpose_bits(chunk_u64, &mut tmp);
                    chunk_u64.copy_from_slice(&tmp);
                }
                BitBuffer::new_with_offset(bytes_mut.freeze().into_byte_buffer(), len, offset)
            }
            Err(bytes) => bits_op_with_copy(bytes, len, offset, fastlanes::transpose_bits),
        }
    } else {
        bits_op_with_copy(bytes, len, offset, fastlanes::transpose_bits)
    }
}

pub fn untranspose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    assert!(
        bits.inner().len().is_multiple_of(128),
        "Transpose BitBuffer byte length must be a multiple of 128"
    );
    assert!(
        bits.inner().is_aligned(Alignment::of::<u64>()),
        "Transposed buffer must be 8 byte aligned"
    );
    let (offset, len, bytes) = bits.into_inner();
    match bytes.try_into_mut() {
        Ok(mut bytes_mut) => {
            let (prefix, middle, trailer) = unsafe { bytes_mut.align_to_mut::<u64>() };
            assert!(
                prefix.is_empty() && trailer.is_empty(),
                "Transposed buffer must be 8 byte aligned"
            );
            let (chunks, _) = middle.as_chunks_mut::<16>();
            let mut tmp = [0u64; 16];
            for chunk in chunks {
                fastlanes::untranspose_bits::<u64>(chunk, &mut tmp);
                chunk.copy_from_slice(&tmp);
            }
            BitBuffer::new_with_offset(bytes_mut.freeze().into_byte_buffer(), len, offset)
        }
        Err(bytes) => bits_op_with_copy(bytes, len, offset, fastlanes::untranspose_bits::<u64>),
    }
}

fn bits_op_with_copy<F: Fn(&[u64; 16], &mut [u64; 16])>(
    bytes: ByteBuffer,
    len: usize,
    offset: usize,
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
        op(
            unsafe { mem::transmute::<&[u8; 128], &[u64; 16]>(input) },
            output,
        );
    }

    if !input_trailer.is_empty() {
        let mut padded_input = [0u8; 128];
        padded_input[0..input_trailer.len()].clone_from_slice(input_trailer);
        op(
            unsafe { mem::transmute::<&[u8; 128], &[u64; 16]>(&padded_input) },
            output_chunks
                .last_mut()
                .vortex_expect("Output wasn't a multiple of 128 bytes"),
        );
    }

    unsafe { output.set_len(output_len) };
    BitBuffer::new_with_offset(
        output.freeze().into_byte_buffer(),
        len.next_multiple_of(1024),
        offset,
    )
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
