use std::ops::Deref;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};

use crate::buffer_mut::{BufferMut, ByteSourceMut};
use crate::{ByteOwner, ByteSource};

unsafe impl ByteSource for Bytes {
    type Owner = Self;

    fn as_bytes(&self) -> &[u8] {
        self.deref()
    }

    fn into_owner(self) -> Self::Owner {
        self
    }

    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>> {
        match Arc::try_unwrap(self) {
            Ok(bytes) => match bytes.try_into_mut() {
                Ok(bytes_mut) => Ok(BufferMut::from_owner(bytes_mut)),
                Err(this_bytes) => Err(Arc::new(this_bytes) as Arc<dyn ByteOwner>),
            },
            Err(this) => Err(this as Arc<dyn ByteOwner>),
        }
    }
}

unsafe impl ByteSourceMut for BytesMut {
    type OwnerMut = Self;

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        BytesMut::as_mut(self)
    }

    fn into_owner(self) -> Self::OwnerMut {
        self
    }
}
