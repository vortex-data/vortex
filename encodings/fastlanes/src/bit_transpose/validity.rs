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
    assert!(
        bits.inner().len().is_multiple_of(128),
        "Transpose BitBuffer must be 128-byte aligned"
    );
    let (offset, len, bytes) = bits.into_inner();
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
    BitBuffer::new_with_offset(
        output.freeze().into_byte_buffer(),
        len.next_multiple_of(1024),
        offset,
    )
}
