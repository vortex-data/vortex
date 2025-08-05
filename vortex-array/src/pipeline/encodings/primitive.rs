// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::N;
use crate::pipeline::bits::BitMask;
use crate::pipeline::buffers::{BufferHandle, ByteBufferHandle};
use crate::pipeline::encodings::{BindContext, Encoding, Evaluation, EvaluationContext};
use crate::pipeline::selection::Selection;
use crate::pipeline::view_mut::ViewMut;
use std::task::{Poll, ready};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;

pub struct PrimitiveEncoding {
    buffer: ByteBufferHandle,
}

impl PrimitiveEncoding {
    pub fn new(buffer: ByteBufferHandle) -> Self {
        Self { buffer }
    }
}

impl Encoding for PrimitiveEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let ptype = ctx.dtype.as_ptype();
        Ok(match_each_native_ptype!(ptype, |T| {
            Box::new(PrimitiveEvaluation::<T> {
                buffer: BufferHandle::from_byte_buffer(self.buffer.clone()),
                len: ctx.len,
                offset: 0,
            }) as Box<dyn Evaluation>
        }))
    }
}

struct PrimitiveEvaluation<T> {
    buffer: BufferHandle<T>,
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
        selected: &dyn BitMask,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        let mut primitive = out.as_primitive::<T>();
        match selected.true_count() {
            0 => {
                // If no elements are selected, we can skip copying.
                self.offset += selected.true_count();
                out.set_selection(Selection::Prefix { len: 0 });
            }
            N => {
                primitive.as_mut()[self.offset..][..N].copy_from_slice(&buffer[self.offset..][..N]);
                self.offset += N;
                out.set_selection(Selection::Prefix { len: N });
            }
            _ => {
                for index in selected.iter_ones() {
                    primitive.as_mut()[self.offset] = buffer[self.offset + index];
                    self.offset += 1;
                }
            }
        }

        out.set_selection_mask(selected);

        Poll::Ready(Ok(()))
    }
}
