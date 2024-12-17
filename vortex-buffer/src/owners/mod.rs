#[cfg(feature = "arrow")]
mod arrow;
#[cfg(feature = "bytes")]
mod bytes;

use std::sync::Arc;

use crate::buffer_mut::{BufferMut, ByteSourceMut};
use crate::{ByteOwner, ByteSource};

unsafe impl ByteSource for &'static [u8] {
    type Owner = Self;

    fn as_bytes(&self) -> &[u8] {
        *self
    }

    fn into_owner(self) -> Self::Owner {
        self
    }

    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>> {
        Err(self)
    }
}

unsafe impl ByteSource for Vec<u8> {
    type Owner = Self;

    fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }

    fn into_owner(self) -> Self::Owner {
        self
    }

    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>> {
        Arc::try_unwrap(self)
            .map(|vec| BufferMut::from_owner(vec))
            .map_err(|this| this as Arc<dyn ByteOwner>)
    }
}

unsafe impl ByteSourceMut for Vec<u8> {
    type OwnerMut = Self;

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }

    fn into_owner(self) -> Self::OwnerMut {
        self
    }
}
