use std::collections::LinkedList;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::{Alignment, BufferMut, ByteBufferMut};

#[derive(Clone, Default, Debug)]
pub struct BufferPool {
    inner: Arc<InnerPool>,
}

#[derive(Debug)]
struct InnerPool {
    buffers: Mutex<LinkedList<ByteBufferMut>>,
    default_capacity: usize,
    default_alignment: Alignment,
}

impl Default for InnerPool {
    fn default() -> Self {
        Self {
            buffers: Default::default(),
            default_capacity: Default::default(),
            default_alignment: Alignment::of::<u64>(),
        }
    }
}

impl InnerPool {
    fn default_buffer(&self) -> ByteBufferMut {
        ByteBufferMut::with_capacity_aligned(self.default_capacity, self.default_alignment)
    }
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_capacity(default_capacity: usize) -> Self {
        let inner = Arc::new(InnerPool {
            default_capacity,
            default_alignment: Alignment::none(),
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
            Some(buffer) => {
                return buffer;
            }
            None => {
                return self.inner.default_buffer();
            }
        }
    }

    pub fn get_aligned<T>(&self) -> BufferMut<T> {
        let buffer = self.get();
        buffer.cast_empty::<T>()
    }

    pub fn put_back<T>(&self, mut buffer: BufferMut<T>) {
        // Safety:
        // This is always a valid state, we just clear the existing data allowing it to be re-used.
        unsafe {
            buffer.set_len(0);
        }
        let buffer = buffer.cast_empty::<u8>();

        // We optimistically try and return the memory
        if let Some(mut pool) = self.inner.buffers.try_lock() {
            pool.push_back(buffer);
        }
    }
}
