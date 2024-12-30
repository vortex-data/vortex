use std::fmt::{Display, Formatter};
use std::ops::Range;

use vortex_error::VortexUnwrap;

#[derive(Copy, Clone, Debug)]
pub struct ByteRange {
    pub begin: u64,
    pub end: u64,
}

impl Display for ByteRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.begin, self.end)
    }
}

impl ByteRange {
    pub fn new(begin: u64, end: u64) -> Self {
        assert!(begin < end, "Buffer begin must be before its end");
        Self { begin, end }
    }

    pub fn len(&self) -> u64 {
        self.end - self.begin
    }

    pub fn is_empty(&self) -> bool {
        self.begin == self.end
    }

    pub fn as_range(&self) -> Range<usize> {
        Range {
            // TODO(ngates): this cast is unsafe and can panic
            start: self.begin.try_into().vortex_unwrap(),
            end: self.end.try_into().vortex_unwrap(),
        }
    }
}
