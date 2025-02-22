use std::collections::LinkedList;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::vortex_panic;

use crate::{Alignment, BufferMut, ByteBufferMut};

#[derive(Clone, Default, Debug)]
pub struct BufferPool {
    inner: Arc<InnerPool>,
}

#[derive(Debug)]
struct InnerPool {
    buffers: Vec<Mutex<LinkedList<ByteBufferMut>>>,
    default_capacity: usize,
}

impl InnerPool {
    fn new(default_capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(8);

        // preallocate some common alignments
        for _ in 0_u8..8 {
            buffers.push(Default::default());
        }

        Self {
            buffers,
            default_capacity,
        }
    }

    fn default_buffer_aligned(&self, alignment: Alignment) -> ByteBufferMut {
        ByteBufferMut::with_capacity_aligned(self.default_capacity, alignment)
    }
}

impl Default for InnerPool {
    fn default() -> Self {
        Self::new(0)
    }
}

impl InnerPool {}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default_capacity(default_capacity: usize) -> Self {
        Self {
            inner: Arc::new(InnerPool::new(default_capacity)),
        }
    }

    pub fn get(&self) -> ByteBufferMut {
        self.get_aligned(Alignment::none())
    }

    pub fn get_aligned(&self, alignment: Alignment) -> ByteBufferMut {
        match self.inner.buffers.get(alignment.exponent() as usize) {
            None => vortex_panic!("oops missing {alignment}"),
            Some(buffer_list) => {
                let mut pool = match buffer_list.try_lock() {
                    Some(pool) => pool,
                    None => {
                        return self.inner.default_buffer_aligned(alignment);
                    }
                };

                match pool.pop_front() {
                    Some(buffer) => buffer,
                    None => self.inner.default_buffer_aligned(alignment),
                }
            }
        }
    }

    pub fn put_back<T>(&self, buffer: BufferMut<T>) {
        // Probably simpler to just allocate
        if buffer.bytes.capacity() == 0 {
            return;
        }

        // we just erase the type info, keeping the alignment
        let buffer = ByteBufferMut {
            bytes: buffer.bytes,
            length: 0,
            alignment: buffer.alignment,
            _marker: std::marker::PhantomData,
        };

        // We optimistically try and return the memory
        match self
            .inner
            .buffers
            .get(buffer.alignment().exponent() as usize)
        {
            None => {}
            Some(pool) => {
                if let Some(mut pool) = pool.try_lock() {
                    pool.push_back(buffer);
                }
            }
        }
    }
}
