use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;

use crate::vtable::{EncodingVTable, VTableRef};
use crate::{ArrayRef, EncodingId};

pub struct ArrayData {
    len: usize,
    dtype: DType,
    encoding: EncodingId,
    metadata: Option<ByteBuffer>,
    buffers: Vec<ByteBuffer>,
    children: Vec<ArrayData>,
}
