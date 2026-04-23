// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder kernel for `Chunked` — send the output builder down to each chunk in turn.

use vortex_error::VortexResult;

use crate::AnyCanonical;
use crate::BuilderStep;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::builder_kernel::AppendToBuilderKernel;
use crate::builders::ArrayBuilder;
use crate::matcher::Matcher;

/// Append a chunked array into a canonical builder, one chunk at a time.
///
/// The kernel uses the chunked array's builder cursor to find the next chunk that has not yet
/// been consumed (i.e. still `Some`). It returns [`crate::BuilderStep::ExecuteSlot`] for that
/// index; the executor drives the chunk to canonical, extends the builder, and nulls the slot.
/// When every chunk slot is `None`, the kernel returns [`crate::BuilderStep::Done`].
#[derive(Debug, Default)]
pub struct ChunkedBuilderKernel;

impl AppendToBuilderKernel<Chunked> for ChunkedBuilderKernel {
    fn append_to_builder(
        &self,
        array: ArrayView<'_, Chunked>,
        builder: Box<dyn ArrayBuilder>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Box<dyn ArrayBuilder>, BuilderStep)> {
        let step = match array.next_builder_slot(array.slots()) {
            Some(idx) => BuilderStep::ExecuteSlot(idx, AnyCanonical::matches),
            None => BuilderStep::Done,
        };
        Ok((builder, step))
    }
}
