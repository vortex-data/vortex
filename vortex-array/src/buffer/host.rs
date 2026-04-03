// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::buffer::BufferTrait;

impl BufferTrait for ByteBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        ByteBuffer::len(self)
    }

    fn alignment(&self) -> Alignment {
        ByteBuffer::alignment(self)
    }

    fn is_on_device(&self) -> bool {
        false
    }

    fn is_on_host(&self) -> bool {
        true
    }

    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        Ok(self.clone().aligned(alignment))
    }

    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        let this = self.clone();
        Ok(Box::pin(async move { Ok(this.aligned(alignment)) }))
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn BufferTrait> {
        Arc::new(ByteBuffer::slice(self, range))
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn BufferTrait>> {
        let this = ByteBuffer::aligned((*self).clone(), alignment);
        Ok(Arc::new(this))
    }
}
