use vortex_buffer::Buffer;
use vortex_error::{VortexExpect, VortexResult};

use crate::visitor::ArrayVisitor;
use crate::ArrayData;

impl ArrayData {
    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> usize {
        let mut visitor = NBytesVisitor(0);
        self.encoding()
            .accept(self.as_ref(), &mut visitor)
            .vortex_expect("Failed to get nbytes from Array");
        visitor.0
    }
}

pub trait ArrayNBytes {
    /// Total size of the array in bytes, including all children and buffers.
    fn nbytes(&self) -> usize;
}

// Implement ArrayNBytes for all concrete arrays.
impl<A: AsRef<ArrayData>> ArrayNBytes for A {
    #[inline(always)]
    fn nbytes(&self) -> usize {
        self.as_ref().nbytes()
    }
}

struct NBytesVisitor(usize);

impl ArrayVisitor for NBytesVisitor {
    fn visit_child(&mut self, _name: &str, array: &ArrayData) -> VortexResult<()> {
        self.0 += array.with_dyn(|a| a.nbytes());
        Ok(())
    }

    fn visit_buffer(&mut self, buffer: &Buffer) -> VortexResult<()> {
        self.0 += buffer.len();
        Ok(())
    }
}
