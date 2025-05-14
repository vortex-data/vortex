use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

use crate::serde::ArrayChildren;
use crate::{Array, ArrayRef, EncodingRef};

#[derive(Clone, Debug)]
pub struct ArrayData {
    encoding: EncodingRef,
    len: usize,
    dtype: DType,
    metadata: Vec<u8>,
    buffers: Vec<ByteBuffer>,
    children: Vec<ArrayData>,
}

impl ArrayChildren for ArrayData {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        if index >= self.children.len() {
            vortex_bail!("Index out of bounds");
        }
        let child = &self.children[index];
        if child.len != len {
            vortex_bail!(
                "Child length mismatch. Provided {}, but actually {}",
                len,
                child.len
            );
        }
        if child.dtype != *dtype {
            vortex_bail!(
                "Child dtype mismatch. Provided {:?}, but actually {:?}",
                dtype,
                child.dtype
            );
        }
        ArrayRef::try_from(child)
    }

    fn len(&self) -> usize {
        self.children.len()
    }
}

impl TryFrom<&dyn Array> for ArrayData {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> Result<Self, Self::Error> {
        Ok(ArrayData {
            encoding: value.encoding(),
            len: value.len(),
            dtype: value.dtype().clone(),
            metadata: value
                .metadata()?
                .ok_or_else(|| vortex_err!("Array does not support serialization"))?,
            buffers: value.buffers().to_vec(),
            children: value
                .children()
                .iter()
                .map(|child| child.as_ref().try_into())
                .try_collect()?,
        })
    }
}

impl TryFrom<&ArrayData> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: &ArrayData) -> Result<Self, Self::Error> {
        value.encoding.build(
            &value.dtype,
            value.len,
            &value.metadata,
            &value.buffers,
            value,
        )
    }
}
