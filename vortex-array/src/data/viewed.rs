use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use flatbuffers::Follow;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::FlatBuffer;

use crate::encoding::opaque::OpaqueEncoding;
use crate::encoding::EncodingRef;
use crate::{flatbuffers as fb, ArrayMetadata, ContextRef};

/// Zero-copy view over flatbuffer-encoded array data, created without eager serialization.
#[derive(Clone)]
pub(super) struct ViewedArrayData {
    pub(super) encoding: EncodingRef,
    pub(super) dtype: DType,
    pub(super) len: usize,
    pub(super) metadata: Arc<dyn ArrayMetadata>,
    pub(super) flatbuffer: FlatBuffer,
    pub(super) flatbuffer_loc: usize,
    pub(super) buffers: Arc<[ByteBuffer]>,
    pub(super) ctx: ContextRef,
    #[cfg(feature = "canonical_counter")]
    pub(super) canonical_counter: Arc<std::sync::atomic::AtomicUsize>,
}

impl Debug for ViewedArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayView")
            .field("encoding", &self.encoding)
            .field("dtype", &self.dtype)
            .field("buffers", &self.buffers)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl ViewedArrayData {
    pub fn flatbuffer(&self) -> fb::Array {
        unsafe { fb::Array::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    pub fn metadata_bytes(&self) -> Option<&[u8]> {
        self.flatbuffer().metadata().map(|m| m.bytes())
    }

    // TODO(ngates): should we separate self and DType lifetimes? Should DType be cloned?
    pub fn child(&self, idx: usize, dtype: &DType, len: usize) -> VortexResult<Self> {
        let child = self
            .array_child(idx)
            .ok_or_else(|| vortex_err!("ArrayView: array_child({idx}) not found"))?;
        let flatbuffer_loc = child._tab.loc();

        let encoding = self
            .ctx
            .lookup_encoding(child.encoding())
            .unwrap_or_else(|| {
                // We must return an EncodingRef, which requires a static reference.
                // OpaqueEncoding however must be created dynamically, since we do not know ahead
                // of time which of the ~65,000 unknown code IDs we will end up seeing. Thus, we
                // allocate (and leak) 2 bytes of memory to create a new encoding.
                Box::leak(Box::new(OpaqueEncoding(child.encoding())))
            });

        let metadata = encoding.load_metadata(child.metadata().map(|m| m.bytes()))?;

        Ok(Self {
            encoding,
            dtype: dtype.clone(),
            len,
            metadata,
            flatbuffer: self.flatbuffer.clone(),
            flatbuffer_loc,
            buffers: self.buffers.clone(),
            ctx: self.ctx.clone(),
            #[cfg(feature = "canonical_counter")]
            canonical_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    fn array_child(&self, idx: usize) -> Option<fb::Array> {
        let children = self.flatbuffer().children()?;
        (idx < children.len()).then(|| children.get(idx))
    }

    pub fn nchildren(&self) -> usize {
        self.flatbuffer().children().map(|c| c.len()).unwrap_or(0)
    }

    pub fn nbuffers(&self) -> usize {
        self.flatbuffer()
            .buffers()
            .map(|b| b.len())
            .unwrap_or_default()
    }

    pub fn buffer(&self, index: usize) -> Option<&ByteBuffer> {
        self.flatbuffer()
            .buffers()
            .map(|buffers| {
                assert!(
                    index < buffers.len(),
                    "ArrayView buffer index out of bounds"
                );
                buffers.get(index) as usize
            })
            .map(|idx| &self.buffers[idx])
    }
}
