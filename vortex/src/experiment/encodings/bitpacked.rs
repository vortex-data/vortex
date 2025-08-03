// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::{
    BindContext, BufferId, Encoding, Evaluation, EvaluationContext,
};
use crate::experiment::mask::BitMask;
use crate::experiment::vector::{BitVector, N, Selection, Vector};
use fastlanes::{BitPacking, FastLanes};
use std::task::{Poll, ready};
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_fastlanes::unpack_iter::BitPacked;
use vortex_mask::Mask;

pub struct BitPackedEncoding {
    bit_width: usize,
    buffer_id: BufferId,
}

impl BitPackedEncoding {
    pub fn new(bit_width: usize, buffer_id: BufferId) -> Self {
        Self {
            bit_width,
            buffer_id,
        }
    }
}

impl Encoding for BitPackedEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let ptype = ctx.dtype.as_ptype();
        match_each_unsigned_integer_ptype!(ptype, |T| {
            Ok(Box::new(BitPackedEvaluation::<T> {
                width: self.bit_width,
                packed_stride: self.bit_width * <T as FastLanes>::LANES,
                buffer_id: self.buffer_id,
                buffer: None,
                packed_offset: 0,
            }))
        })
    }
}

struct BitPackedEvaluation<T> {
    width: usize,
    packed_stride: usize,

    buffer_id: BufferId,
    buffer: Option<Buffer<T>>,
    packed_offset: usize,
}

impl<T: NativePType + BitPacking> Evaluation for BitPackedEvaluation<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &BitMask,
        defined: &BitMask,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>> {
        if self.buffer.is_none() {
            let byte_buffer = ready!(ctx.buffer(self.buffer_id))?;
            self.buffer = Some(Buffer::<T>::from_byte_buffer(byte_buffer));
        };
        let buffer = self.buffer.as_ref().vortex_expect("Infallible");

        if selected.true_count() < 16 {
            // TODO(ngates): I think we found it was <= 8 elements where unpack_single is faster
            //  than unpacking the whole chunk... Given we do two chunks, that's ~16 elements?
            //  We could also intersect with the defined mask to see if we can prune further, but
            //  may not be worth doing the extra work.
        }

        // Otherwise, we unconditionally unpack two chunks of 1024 elements each into the
        // output vector, and simply return the mask we were given.
        let mut view = out.as_primitive::<T>();
        let packed = &buffer.as_slice()[self.packed_offset..];
        for i in 0..(N / 1024) {
            unsafe {
                BitPacking::unchecked_unpack(
                    self.width,
                    &packed[(i * self.packed_stride)..][..self.packed_stride],
                    &mut view.as_mut()[(i * 1024)..][..1024],
                );
            }
        }

        self.packed_offset += (N / 1024) * self.packed_stride;

        // Set the selection to the given mask, which is a bit array of length N.
        out.set_selection_mask(selected.clone());

        Poll::Ready(Ok(()))
    }
}
