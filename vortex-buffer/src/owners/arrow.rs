use std::io;
use std::sync::Arc;

use crate::buffer_mut::{BufferMut, ByteSourceMut};
use crate::{Buffer, ByteOwner, ByteSource};

unsafe impl ByteSource for arrow_buffer::Buffer {
    type Owner = Self;

    fn as_bytes(&self) -> &[u8] {
        self.as_slice()
    }

    fn into_owner(self) -> Self::Owner {
        self
    }

    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>> {
        match Arc::try_unwrap(self) {
            Ok(buffer) => match buffer.into_mutable() {
                Ok(mut_buffer) => Ok(BufferMut::from_owner(mut_buffer)),
                Err(this_buffer) => Err(Arc::new(this_buffer) as Arc<dyn ByteOwner>),
            },
            Err(this) => Err(this as Arc<dyn ByteOwner>),
        }
    }
}

unsafe impl ByteSourceMut for arrow_buffer::MutableBuffer {
    type OwnerMut = Self;

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        self.as_mut()
    }

    fn into_owner(self) -> Self::OwnerMut {
        self
    }
}

impl Buffer {
    pub fn into_arrow(self) -> arrow_buffer::Buffer {
        match self.into_owner::<arrow_buffer::Buffer>() {
            None => {}
            Some(arrow_buffer) => arrow_buffer,
        }
    }
}
