use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;

use crate::array::ArrayRef;

pub struct ArrayData {
    len: usize,
    dtype: DType,
    metadata: Option<ByteBuffer>,
    buffers: Vec<ByteBuffer>,
    children: Vec<ArrayRef>,
}
