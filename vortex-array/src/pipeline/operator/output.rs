// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{BoolArray, PrimitiveArray};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, N};
use crate::validity::Validity;
use crate::Canonical;
use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, BufferMut, ByteBuffer};
use vortex_dtype::NativePType;
use vortex_error::VortexResult;

/// This trait allows us to abstract over the canonical element type of the pipeline, providing
/// a single implementation of the pipeline batch execution for all canonical types.
pub trait PipelineOutput: Send {
    type Element: Element;
    fn allocate(capacity: usize) -> Self;
    fn view_mut(&mut self, offset: usize) -> ViewMut<'_>;
    fn into_canonical(self, len: usize) -> VortexResult<Canonical>;
}

pub(super) struct BoolOutput {
    buffer: BufferMut<bool>,
}

impl PipelineOutput for BoolOutput {
    type Element = bool;

    fn allocate(capacity: usize) -> Self {
        let mut buffer = BufferMut::with_capacity(capacity);
        unsafe { buffer.set_len(capacity) };
        BoolOutput { buffer }
    }

    fn view_mut(&mut self, offset: usize) -> ViewMut<'_> {
        ViewMut::new(&mut self.buffer[offset..][..N], None)
    }

    fn into_canonical(mut self, len: usize) -> VortexResult<Canonical> {
        unsafe { self.buffer.set_len(len) };

        let buffer = ByteBuffer::from_arrow_buffer(
            BooleanBuffer::from(self.buffer.as_ref()).into_inner(),
            Alignment::of::<u64>(),
        );

        Ok(Canonical::Bool(BoolArray::try_new(
            buffer,
            0,
            len,
            Validity::NonNullable,
        )?))
    }
}

pub(super) struct PrimitiveOutput<T> {
    buffer: BufferMut<T>,
}

impl<T: NativePType + Element> PipelineOutput for PrimitiveOutput<T> {
    type Element = T;

    fn allocate(capacity: usize) -> Self {
        let mut buffer = BufferMut::with_capacity(capacity);
        unsafe { buffer.set_len(capacity) };
        PrimitiveOutput { buffer }
    }

    fn view_mut(&mut self, offset: usize) -> ViewMut<'_> {
        ViewMut::new(&mut self.buffer[offset..][..N], None)
    }

    fn into_canonical(mut self, len: usize) -> VortexResult<Canonical> {
        unsafe { self.buffer.set_len(len) };
        Ok(Canonical::Primitive(PrimitiveArray::new(
            self.buffer.freeze(),
            Validity::NonNullable,
        )))
    }
}
