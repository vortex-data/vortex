use std::collections::LinkedList;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::{BufferMut, ByteBufferMut};

#[derive(Clone, Default, Debug)]
pub struct BufferPool {
    inner: Arc<InnerPool>,
}

#[derive(Debug)]
struct InnerPool {
    buffers: Mutex<LinkedList<ByteBufferMut>>,
    default_capacity: usize,
}

impl Default for InnerPool {
    fn default() -> Self {
        Self {
            buffers: Default::default(),
            default_capacity: Default::default(),
        }
    }
}

impl InnerPool {
    fn default_buffer(&self) -> ByteBufferMut {
        ByteBufferMut::with_capacity(self.default_capacity)
    }
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_capacity(default_capacity: usize) -> Self {
        let inner = Arc::new(InnerPool {
            default_capacity,
            buffers: Default::default(),
        });

        Self { inner }
    }

    pub fn get(&self) -> ByteBufferMut {
        let mut pool = match self.inner.buffers.try_lock() {
            Some(pool) => pool,
            None => {
                return self.inner.default_buffer();
            }
        };

        match pool.pop_front() {
            Some(buffer) => buffer,
            None => self.inner.default_buffer(),
        }
    }

    pub fn get_aligned<T>(&self) -> BufferMut<T> {
        let buffer = self.get();
        buffer.cast_empty::<T>()
    }

    pub fn put_back<T>(&self, buffer: BufferMut<T>) {
        // we just erase the type info, keeping the alignment
        let buffer = ByteBufferMut {
            bytes: buffer.bytes,
            length: 0,
            alignment: buffer.alignment,
            _marker: std::marker::PhantomData,
        };

        // We optimistically try and return the memory
        if let Some(mut pool) = self.inner.buffers.try_lock() {
            pool.push_back(buffer);
        }
    }
}
