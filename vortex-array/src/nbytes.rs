use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};

use crate::visitor::ArrayVisitor;
use crate::ArrayData;

impl ArrayData {
    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> usize {
        let mut visitor = NBytesVisitor::default();
        self.encoding()
            .accept(self.as_ref(), &mut visitor)
            .vortex_expect("Failed to get nbytes from Array");
        visitor.0 + self.metadata_bytes().map_or(0, |b| b.len())
    }
}

#[derive(Default)]
struct NBytesVisitor(usize);

impl ArrayVisitor for NBytesVisitor {
    fn visit_child(&mut self, _name: &str, array: &ArrayData) -> VortexResult<()> {
        self.0 += array.nbytes();
        Ok(())
    }

    fn visit_buffer(&mut self, buffer: &ByteBuffer) -> VortexResult<()> {
        self.0 += buffer.len();
        Ok(())
    }
}
