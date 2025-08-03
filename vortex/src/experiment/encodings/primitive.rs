// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::{
    BindContext, BufferId, Encoding, Evaluation, EvaluationContext,
};
use crate::experiment::mask::BitMask;
use crate::experiment::vector::{BitVector, N, Vector};
use std::task::{Poll, ready};
use vortex_array::stats::StatsSet;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};

pub struct PrimitiveEncoding {
    buffer_id: BufferId,
}

impl Encoding for PrimitiveEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let ptype = ctx.dtype.as_ptype();
        Ok(match_each_native_ptype!(ptype, |T| {
            Box::new(PrimitiveEvaluation::<T> {
                buffer_id: self.buffer_id,
                len: ctx.len,
                offset: 0,
                buffer: None,
            }) as Box<dyn Evaluation>
        }))
    }
}

struct PrimitiveEvaluation<T> {
    buffer_id: BufferId,
    // The source buffer.
    buffer: Option<Buffer<T>>,
    // The overall length of the data.
    len: usize,
    // The current row offset.
    offset: usize,
}

impl<T: NativePType> Evaluation for PrimitiveEvaluation<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * N;
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

        let mut primitive = out.as_primitive::<T>();
        match selected {
            BitMask::All => {
                primitive.as_mut()[self.offset..][..N].copy_from_slice(&buffer[self.offset..][..N]);
                self.offset += N;
            }
            BitMask::None => {}
            BitMask::Some(indices) => {
                for index in indices.iter_ones() {
                    primitive.as_mut()[self.offset] = buffer[self.offset + index];
                    self.offset += 1;
                }
            }
        }

        Poll::Ready(Ok(()))
    }
}
