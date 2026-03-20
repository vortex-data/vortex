// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::mem::MaybeUninit;

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::bit_transpose::transpose_bits;
use crate::bit_transpose::untranspose_bits;

pub fn transpose_validity(validity: &Validity, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
    match validity {
        Validity::Array(mask) => {
            let bools = mask
                .clone()
                .execute::<Canonical>(ctx)?
                .into_bool()
                .into_bit_buffer();

            Ok(Validity::Array(
                BoolArray::new(transpose_bitbuffer(bools), Validity::NonNullable).into_array(),
            ))
        }
        v @ Validity::AllValid | v @ Validity::AllInvalid | v @ Validity::NonNullable => {
            Ok(v.clone())
        }
    }
}

#[inline]
pub fn transpose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    let (offset, len, bytes) = bits.into_inner();

    if bytes.len().is_multiple_of(128) {
        match bytes.try_into_mut() {
            Ok(mut bytes_mut) => {
                // We can ignore the spare trailer capacity that can be an artifact of allocator as we requested 128 multiple chunks
                let (chunks, _) = bytes_mut.as_chunks_mut::<128>();
                let mut tmp = [0u8; 128];
                for chunk in chunks {
                    transpose_bits(chunk, &mut tmp);
                    chunk.copy_from_slice(&tmp);
                }
                BitBuffer::new_with_offset(bytes_mut.freeze().into_byte_buffer(), len, offset)
            }
            Err(bytes) => bits_op_with_copy(bytes, len, offset, transpose_bits),
        }
    } else {
        bits_op_with_copy(bytes, len, offset, transpose_bits)
    }
}

pub fn untranspose_validity(validity: &Validity, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
    match validity {
        Validity::Array(mask) => {
            let bools = mask
                .clone()
                .execute::<Canonical>(ctx)?
                .into_bool()
                .into_bit_buffer();

            Ok(Validity::Array(
                BoolArray::new(untranspose_bitbuffer(bools), Validity::NonNullable).into_array(),
            ))
        }
        v @ Validity::AllValid | v @ Validity::AllInvalid | v @ Validity::NonNullable => {
            Ok(v.clone())
        }
    }
}

#[inline]
pub fn untranspose_bitbuffer(bits: BitBuffer) -> BitBuffer {
    let (offset, len, bytes) = bits.into_inner();

    if bytes.len().is_multiple_of(128) {
        match bytes.try_into_mut() {
            Ok(mut bytes_mut) => {
                let (chunks, _) = bytes_mut.as_chunks_mut::<128>();
                let mut tmp = [0u8; 128];
                for chunk in chunks {
                    untranspose_bits(chunk, &mut tmp);
                    chunk.copy_from_slice(&tmp);
                }
                BitBuffer::new_with_offset(bytes_mut.freeze().into_byte_buffer(), len, offset)
            }
            Err(bytes) => bits_op_with_copy(bytes, len, offset, untranspose_bits),
        }
    } else {
        bits_op_with_copy(bytes, len, offset, untranspose_bits)
    }
}

fn bits_op_with_copy<F: Fn(&[u8; 128], &mut [u8; 128])>(
    bytes: ByteBuffer,
    len: usize,
    offset: usize,
    op: F,
) -> BitBuffer {
    let output_len = bytes.len().next_multiple_of(128);
    let mut output = ByteBufferMut::with_capacity(output_len);
    let (input_chunks, input_trailer) = bytes.as_chunks::<128>();
    // We can ignore the spare trailer capacity that can be an artifact of allocator as we requested 128 multiple chunks
    let (output_chunks, _) = output.spare_capacity_mut().as_chunks_mut::<128>();

    for (input, output) in input_chunks.iter().zip(output_chunks.iter_mut()) {
        op(input, unsafe {
            mem::transmute::<&mut [MaybeUninit<u8>; 128], &mut [u8; 128]>(output)
        });
    }

    if !input_trailer.is_empty() {
        let mut padded_input = [0u8; 128];
        padded_input[0..input_trailer.len()].clone_from_slice(input_trailer);
        op(&padded_input, unsafe {
            mem::transmute::<&mut [MaybeUninit<u8>; 128], &mut [u8; 128]>(
                output_chunks
                    .last_mut()
                    .vortex_expect("Output wasn't a multiple of 128 bytes"),
            )
        });
    }

    unsafe { output.set_len(output_len) };
    BitBuffer::new_with_offset(output.freeze().into_byte_buffer(), len, offset)
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::validity::Validity;
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
    fn transpose_roundtrip_preserves_len_inplace_path() {
        let bits = make_validity_bits(1024);
        assert_eq!(bits.len(), 1024);

        let transposed = transpose_bitbuffer(bits.clone());
        assert_eq!(transposed.len(), 1024);

        let roundtripped = untranspose_bitbuffer(transposed);
        assert_eq!(roundtripped.len(), 1024);
        assert_eq!(roundtripped, bits);
    }

    #[test]
    fn transpose_roundtrip_preserves_len_copy_path() {
        let bits = make_validity_bits(1024);
        let (bits_shared, _hold) = force_copy_path(bits.clone());

        let transposed = transpose_bitbuffer(bits_shared);
        assert_eq!(transposed.len(), 1024);

        let roundtripped = untranspose_bitbuffer(transposed);
        assert_eq!(roundtripped.len(), 1024);
        assert_eq!(roundtripped, bits);
    }

    #[test]
    fn transpose_preserves_len_non_aligned_copy_path() {
        let bits = make_validity_bits(500);
        assert_eq!(bits.len(), 500);

        let transposed = transpose_bitbuffer(bits);
        assert_eq!(transposed.len(), 500);
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
    fn transpose_validity_roundtrip_non_aligned() {
        let bits = make_validity_bits(1500);
        let validity = Validity::Array(BoolArray::new(bits, Validity::NonNullable).into_array());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let transposed = transpose_validity(&validity, &mut ctx).unwrap();

        if let Validity::Array(arr) = &transposed {
            assert_eq!(arr.len(), 1500);
        } else {
            panic!("expected Validity::Array");
        }

        let roundtripped = untranspose_validity(&transposed, &mut ctx).unwrap();
        if let Validity::Array(arr) = &roundtripped {
            assert_eq!(arr.len(), 1500);
        } else {
            panic!("expected Validity::Array");
        }
    }
}
