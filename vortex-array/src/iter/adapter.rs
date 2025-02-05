use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::iter::ArrayIterator;
use crate::Array;

pub struct ArrayIteratorAdapter<I> {
    dtype: DType,
    inner: I,
}

impl<I> ArrayIteratorAdapter<I> {
    pub fn new(dtype: DType, inner: I) -> Self {
        Self { dtype, inner }
    }
}

impl<I> Iterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<Array>>,
{
    type Item = VortexResult<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I> ArrayIterator for ArrayIteratorAdapter<I>
where
    I: Iterator<Item = VortexResult<Array>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}
