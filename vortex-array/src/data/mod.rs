use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::future::ready;
use std::sync::{Arc, RwLock};

use itertools::Itertools;
use owned::OwnedArrayData;
use viewed::ViewedArrayData;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};

use crate::encoding::{EncodingId, EncodingRef};
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::stats::StatsSet;
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::{ArrayChildrenIterator, ArrayDType, ArrayMetadata, ArrayTrait, Context};

mod owned;
mod viewed;

/// A central type for all Vortex arrays, which are known length sequences of typed and possibly compressed data.
///
/// This is the main entrypoint for working with in-memory Vortex data, and dispatches work over the underlying encoding or memory representations.
#[derive(Debug, Clone)]
pub struct ArrayData(pub(crate) InnerArrayData);

// TODO(ngates): make this non-pub once TypedArray disappears
#[derive(Debug, Clone)]
pub(crate) enum InnerArrayData {
    /// Owned [`ArrayData`] with serialized metadata, backed by heap-allocated memory.
    Owned(OwnedArrayData),
    /// Zero-copy view over flatbuffer-encoded [`ArrayData`] data, created without eager serialization.
    Viewed(ViewedArrayData),
}

impl ArrayData {
    pub fn try_new_owned(
        encoding: EncodingRef,
        dtype: DType,
        len: usize,
        metadata: Arc<dyn ArrayMetadata>,
        buffer: Option<Buffer>,
        children: Arc<[ArrayData]>,
        statistics: StatsSet,
    ) -> VortexResult<Self> {
        let data = OwnedArrayData {
            encoding,
            dtype,
            len,
            metadata,
            buffer,
            children,
            stats_map: Arc::new(RwLock::new(statistics)),
        };

        let array = ArrayData(InnerArrayData::Owned(data));
        // Validate here that the metadata correctly parses, so that an encoding can infallibly
        // FIXME(robert): Encoding::with_dyn no longer eagerly validates metadata, come up with a way to validate metadata
        encoding.with_dyn(&array, &mut |_| Ok(()))?;

        Ok(array)
    }

    pub fn try_new_viewed<F>(
        ctx: Arc<Context>,
        dtype: DType,
        len: usize,
        flatbuffer: Buffer,
        flatbuffer_init: F,
        buffers: Vec<Buffer>,
    ) -> VortexResult<Self>
    where
        F: FnOnce(&[u8]) -> VortexResult<crate::flatbuffers::Array>,
    {
        let array = flatbuffer_init(flatbuffer.as_ref())?;
        let flatbuffer_loc = array._tab.loc();

        let encoding = ctx.lookup_encoding(array.encoding()).ok_or_else(
            || {
                let pretty_known_encodings = ctx.encodings()
                    .format_with("\n", |e, f| f(&format_args!("- {}", e.id())));
                vortex_err!(InvalidSerde: "Unknown encoding with ID {:#02x}. Known encodings:\n{pretty_known_encodings}", array.encoding())
            },
        )?;

        let view = ViewedArrayData {
            encoding,
            dtype,
            len,
            flatbuffer,
            flatbuffer_loc,
            buffers: buffers.into(),
            ctx,
        };

        // Validate here that the metadata correctly parses, so that an encoding can infallibly
        // implement Encoding::with_view().
        // FIXME(ngates): validate the metadata
        ArrayData::from(view.clone()).with_dyn(|_| Ok::<(), VortexError>(()))?;

        Ok(view.into())
    }

    pub fn encoding(&self) -> EncodingRef {
        match &self.0 {
            InnerArrayData::Owned(d) => d.encoding(),
            InnerArrayData::Viewed(v) => v.encoding(),
        }
    }

    /// Returns the number of logical elements in the array.
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        match &self.0 {
            InnerArrayData::Owned(d) => d.len(),
            InnerArrayData::Viewed(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.0 {
            InnerArrayData::Owned(d) => d.is_empty(),
            InnerArrayData::Viewed(v) => v.is_empty(),
        }
    }

    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> usize {
        self.with_dyn(|a| a.nbytes())
    }

    pub fn child<'a>(&'a self, idx: usize, dtype: &'a DType, len: usize) -> VortexResult<Self> {
        match &self.0 {
            InnerArrayData::Owned(d) => d.child(idx, dtype, len).cloned(),
            InnerArrayData::Viewed(v) => v
                .child(idx, dtype, len)
                .map(|view| ArrayData(InnerArrayData::Viewed(view))),
        }
    }

    /// Returns a Vec of Arrays with all the array's child arrays.
    pub fn children(&self) -> Vec<ArrayData> {
        match &self.0 {
            InnerArrayData::Owned(d) => d.children().iter().cloned().collect_vec(),
            InnerArrayData::Viewed(v) => v.children(),
        }
    }

    /// Returns the number of child arrays
    pub fn nchildren(&self) -> usize {
        match &self.0 {
            InnerArrayData::Owned(d) => d.nchildren(),
            InnerArrayData::Viewed(v) => v.nchildren(),
        }
    }

    pub fn depth_first_traversal(&self) -> ArrayChildrenIterator {
        ArrayChildrenIterator::new(self.clone())
    }

    /// Count the number of cumulative buffers encoded by self.
    pub fn cumulative_nbuffers(&self) -> usize {
        self.children()
            .iter()
            .map(|child| child.cumulative_nbuffers())
            .sum::<usize>()
            + if self.buffer().is_some() { 1 } else { 0 }
    }

    /// Return the buffer offsets and the total length of all buffers, assuming the given alignment.
    /// This includes all child buffers.
    pub fn all_buffer_offsets(&self, alignment: usize) -> Vec<u64> {
        let mut offsets = vec![];
        let mut offset = 0;

        for col_data in self.depth_first_traversal() {
            if let Some(buffer) = col_data.buffer() {
                offsets.push(offset as u64);

                let buffer_size = buffer.len();
                let aligned_size = (buffer_size + (alignment - 1)) & !(alignment - 1);
                offset += aligned_size;
            }
        }
        offsets.push(offset as u64);

        offsets
    }

    /// Get back the (possibly owned) metadata for the array.
    ///
    /// View arrays will return a reference to their bytes, while heap-backed arrays
    /// must first serialize their metadata, returning an owned byte array to the caller.
    pub fn metadata(&self) -> VortexResult<Cow<[u8]>> {
        match &self.0 {
            InnerArrayData::Owned(array_data) => {
                // Heap-backed arrays must first try and serialize the metadata.
                let owned_meta: Vec<u8> = array_data
                    .metadata()
                    .try_serialize_metadata()?
                    .as_ref()
                    .to_owned();

                Ok(Cow::Owned(owned_meta))
            }
            InnerArrayData::Viewed(array_view) => {
                // View arrays have direct access to metadata bytes.
                array_view
                    .metadata()
                    .ok_or_else(|| vortex_err!("things"))
                    .map(Cow::Borrowed)
            }
        }
    }

    pub fn buffer(&self) -> Option<&Buffer> {
        match &self.0 {
            InnerArrayData::Owned(d) => d.buffer(),
            InnerArrayData::Viewed(v) => v.buffer(),
        }
    }

    pub fn into_buffer(self) -> Option<Buffer> {
        match self.0 {
            InnerArrayData::Owned(d) => d.into_buffer(),
            InnerArrayData::Viewed(v) => v.buffer().cloned(),
        }
    }

    pub fn into_array_iterator(self) -> impl ArrayIterator {
        ArrayIteratorAdapter::new(self.dtype().clone(), std::iter::once(Ok(self)))
    }

    pub fn into_array_stream(self) -> impl ArrayStream {
        ArrayStreamAdapter::new(
            self.dtype().clone(),
            futures_util::stream::once(ready(Ok(self))),
        )
    }

    /// Checks whether array is of a given encoding.
    pub fn is_encoding(&self, id: EncodingId) -> bool {
        self.encoding().id() == id
    }

    #[inline]
    pub fn with_dyn<R, F>(&self, mut f: F) -> R
    where
        F: FnMut(&dyn ArrayTrait) -> R,
    {
        let mut result = None;

        self.encoding()
            .with_dyn(self, &mut |array| {
                // Sanity check that the encoding implements the correct array trait
                debug_assert!(
                    match array.dtype() {
                        DType::Null => array.as_null_array().is_some(),
                        DType::Bool(_) => array.as_bool_array().is_some(),
                        DType::Primitive(..) => array.as_primitive_array().is_some(),
                        DType::Utf8(_) => array.as_utf8_array().is_some(),
                        DType::Binary(_) => array.as_binary_array().is_some(),
                        DType::Struct(..) => array.as_struct_array().is_some(),
                        DType::List(..) => array.as_list_array().is_some(),
                        DType::Extension(..) => array.as_extension_array().is_some(),
                    },
                    "Encoding {} does not implement the variant trait for {}",
                    self.encoding().id(),
                    array.dtype()
                );

                result = Some(f(array));
                Ok(())
            })
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to convert Array to {}",
                    std::any::type_name::<dyn ArrayTrait>()
                )
            });

        // Now we unwrap the optional, which we know to be populated by the closure.
        result.vortex_expect("Failed to get result from Array::with_dyn")
    }
}

impl Display for ArrayData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = match &self.0 {
            InnerArrayData::Owned(_) => "",
            InnerArrayData::Viewed(_) => "$",
        };
        write!(
            f,
            "{}{}({}, len={})",
            prefix,
            self.encoding().id(),
            self.dtype(),
            self.len()
        )
    }
}
