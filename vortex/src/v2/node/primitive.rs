// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::vector::{N, Vector};
use crate::v2::node::{BindContext, BufferId, Evaluation, EvaluationContext, Node};
use std::task::Poll;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};

#[macro_export]
macro_rules! ready {
    ($e:expr) => {
        match $e {
            std::task::Poll::Ready(t) => t,
            std::task::Poll::Pending => {
                return Ok(std::task::Poll::Pending);
            }
        }
    };
}

pub struct PrimitiveNode {
    buffer_id: BufferId,
}

struct PrimitiveEvaluation<T> {
    buffer_id: BufferId,
    // The current row offset.
    offset: usize,
    // The source buffer.
    buffer: Option<Buffer<T>>,
}

impl<T: NativePType> Evaluation for PrimitiveEvaluation<T> {
    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &Mask,
        defined: &Mask,
        out: &mut Vector,
    ) -> VortexResult<Poll<()>> {
        if self.buffer.is_none() {
            match ctx.get_buffer(self.buffer_id)? {
                None => return Ok(Poll::Pending),
                Some(buffer) => {
                    self.buffer = Some(Buffer::<T>::from_byte_buffer(buffer));
                }
            }
        };

        let buffer = self.buffer.as_ref().vortex_expect("Infallible");

        let mut primitive = out.as_primitive::<T>();

        // TODO(ngates): should we actually just iterate over the selected mask?
        match selected.indices() {
            AllOr::All => {
                primitive.as_mut()[self.offset..][..N].copy_from_slice(&buffer[self.offset..][..N]);
                self.offset += N;
            }
            AllOr::None => {}
            AllOr::Some(indices) => {
                for &index in indices {
                    primitive.as_mut()[self.offset] = buffer[self.offset + index];
                    self.offset += 1;
                }
            }
        }

        Ok(Poll::Ready(()))
    }
}
