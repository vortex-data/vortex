use std::fmt::{Display, Formatter};
use std::sync::{Arc, RwLock};

use owned::OwnedArray;
use viewed::ViewedArray;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_flatbuffers::FlatBuffer;
use vortex_scalar::Scalar;

use crate::array::{
    BoolEncoding, ChunkedArray, ExtensionEncoding, ListEncoding, NullEncoding, PrimitiveEncoding,
    StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::compute::scalar_at;
use crate::encoding::{Encoding, EncodingId};
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::stats::{Stat, StatsSet};
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{ArrayChildrenIterator, ChildrenCollector, ContextRef, NamedChildrenCollector};

mod owned;
mod statistics;
mod viewed;

/// A central type for all Vortex arrays, which are known length sequences of typed and possibly compressed data.
///
/// This is the main entrypoint for working with in-memory Vortex data, and dispatches work over the underlying encoding or memory representations.
#[derive(Debug, Clone)]
pub struct Array(InnerArray);

#[derive(Debug, Clone)]
enum InnerArray {
    /// Owned [`Array`] with serialized metadata, backed by heap-allocated memory.
    Owned(Arc<OwnedArray>),
    /// Zero-copy view over flatbuffer-encoded [`Array`] data, created without eager serialization.
    Viewed(ViewedArray),
}

impl From<OwnedArray> for Array {
    fn from(data: OwnedArray) -> Self {
        Array(InnerArray::Owned(Arc::new(data)))
    }
}

impl From<ViewedArray> for Array {
    fn from(data: ViewedArray) -> Self {
        Array(InnerArray::Viewed(data))
    }
}

impl Array {
    pub fn try_new_owned(
        encoding: VTableRef,
        dtype: DType,
        len: usize,
        metadata: Option<ByteBuffer>,
        buffers: Option<Box<[ByteBuffer]>>,
        children: Option<Box<[Array]>>,
        statistics: StatsSet,
    ) -> VortexResult<Self> {
        Self::try_new(InnerArray::Owned(Arc::new(OwnedArray {
            encoding,
            dtype,
            len,
            metadata,
            buffers,
            children,
            stats_set: RwLock::new(statistics),
            #[cfg(feature = "canonical_counter")]
            canonical_counter: std::sync::atomic::AtomicUsize::new(0),
        })))
    }

    pub fn try_new_viewed<F>(
        ctx: ContextRef,
        dtype: DType,
        len: usize,
        flatbuffer: FlatBuffer,
        flatbuffer_init: F,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<Self>
    where
        F: FnOnce(&[u8]) -> VortexResult<crate::flatbuffers::ArrayNode>,
    {
        let array = flatbuffer_init(flatbuffer.as_ref())?;
        let flatbuffer_loc = array._tab.loc();
        let encoding = ctx.lookup_encoding_or_opaque(array.encoding());

        let view = ViewedArray {
            encoding,
            dtype,
            len,
            flatbuffer,
            flatbuffer_loc,
            buffers: buffers.into(),
            ctx,
            #[cfg(feature = "canonical_counter")]
            canonical_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };

        Self::try_new(InnerArray::Viewed(view))
    }

    /// Shared constructor that performs common array validation.
    fn try_new(inner: InnerArray) -> VortexResult<Self> {
        let array = Array(inner);

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
            array.encoding(),
            array.dtype()
        );

        // First, we validate the metadata.
        array.vtable().validate_metadata(array.metadata_bytes())?;
        // Then perform additional custom validation
        // This is called for both Owned and Viewed array data since there are public functions
        // for constructing an Array, e.g. `try_new_owned`.
        array.vtable().validate(&array)?;

        // Validate that the ArrayVisitor correctly returns the number of buffers and children
        #[cfg(debug_assertions)]
        {
            use crate::visitor::ArrayVisitor;

            #[derive(Default)]
            struct CountVisitor {
                nbuffers: usize,
                nchildren: usize,
            }

            impl ArrayVisitor for CountVisitor {
                fn visit_child(&mut self, _name: &str, _array: &Array) -> VortexResult<()> {
                    self.nchildren += 1;
                    Ok(())
                }

                fn visit_buffer(&mut self, _buffer: &ByteBuffer) -> VortexResult<()> {
                    self.nbuffers += 1;
                    Ok(())
                }
            }

            let mut visitor = CountVisitor::default();
            array.vtable().accept(&array, &mut visitor)?;

            assert_eq!(
                visitor.nbuffers,
                array.nbuffers(),
                "Array visitor gave {} buffers, but Array has {} buffers, {}",
                visitor.nbuffers,
                array.nbuffers(),
                array.encoding(),
            );
            assert_eq!(
                visitor.nchildren,
                array.nchildren(),
                "Array visitor gave {} children, but Array has {} children, {}",
                visitor.nchildren,
                array.nchildren(),
                array.encoding(),
            );
        }

        Ok(array)
    }

    /// Return the array's encoding VTable.
    pub fn vtable(&self) -> &VTableRef {
        match &self.0 {
            InnerArray::Owned(d) => &d.encoding,
            InnerArray::Viewed(v) => &v.encoding,
        }
    }

    /// Return the array's encoding ID.
    pub fn encoding(&self) -> EncodingId {
        self.vtable().id()
    }

    /// Returns the number of logical elements in the array.
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        match &self.0 {
            InnerArray::Owned(d) => d.len,
            InnerArray::Viewed(v) => v.len,
        }
    }

    /// Check whether the array has any data
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the array's dtype
    pub fn dtype(&self) -> &DType {
        match &self.0 {
            InnerArray::Owned(d) => &d.dtype,
            InnerArray::Viewed(v) => &v.dtype,
        }
    }

    /// Whether the array is of a canonical encoding.
    pub fn is_canonical(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(StructEncoding.id())
            || self.is_encoding(ListEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
            || self.is_encoding(ExtensionEncoding.id())
    }

    /// Whether the array is fully zero-copy to Arrow (including children).
    /// This means any nested types, like Structs, Lists, and Extensions are not present.
    pub fn is_arrow(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(VarBinEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
    }

    /// Return whether the array is constant.
    pub fn is_constant(&self) -> bool {
        self.statistics()
            .compute_as::<bool>(Stat::IsConstant)
            .unwrap_or(false)
    }

    /// Return scalar value of this array if the array is constant
    pub fn as_constant(&self) -> Option<Scalar> {
        self.is_constant()
            // This is safe to unwrap as long as empty arrays aren't constant
            .then(|| scalar_at(self, 0).vortex_expect("expected a scalar value"))
    }

    pub fn child<'a>(&'a self, idx: usize, dtype: &'a DType, len: usize) -> VortexResult<Self> {
        match &self.0 {
            InnerArray::Owned(d) => d.child(idx, dtype, len).cloned(),
            InnerArray::Viewed(v) => v
                .child(idx, dtype, len)
                .map(|view| Array(InnerArray::Viewed(view))),
        }
    }

    /// Returns a Vec of Arrays with all the array's child arrays.
    // TODO(ngates): deprecate this function and return impl Iterator
    pub fn children(&self) -> Vec<Array> {
        match &self.0 {
            InnerArray::Owned(d) => d.children.as_ref().map(|c| c.to_vec()).unwrap_or_default(),
            InnerArray::Viewed(_) => {
                let mut collector = ChildrenCollector::default();
                self.vtable()
                    .accept(self, &mut collector)
                    .vortex_expect("Failed to get children");
                collector.children()
            }
        }
    }

    /// Returns a Vec of Arrays with all the array's child arrays.
    pub fn named_children(&self) -> Vec<(String, Array)> {
        let mut collector = NamedChildrenCollector::default();
        self.vtable()
            .accept(&self.clone(), &mut collector)
            .vortex_expect("Failed to get children");
        collector.children()
    }

    /// Returns the number of child arrays
    pub fn nchildren(&self) -> usize {
        match &self.0 {
            InnerArray::Owned(d) => d.nchildren(),
            InnerArray::Viewed(v) => v.nchildren(),
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
            + self.nbuffers()
    }

    /// Return the buffer offsets and the total length of all buffers, assuming the given alignment.
    /// This includes all child buffers.
    pub fn all_buffer_offsets(&self, alignment: usize) -> Vec<u64> {
        let mut offsets = vec![];
        let mut offset = 0;

        for col_data in self.depth_first_traversal() {
            for buffer in col_data.byte_buffers() {
                offsets.push(offset as u64);

                let buffer_size = buffer.len();
                let aligned_size = (buffer_size + (alignment - 1)) & !(alignment - 1);
                offset += aligned_size;
            }
        }
        offsets.push(offset as u64);

        offsets
    }

    pub fn metadata_bytes(&self) -> Option<&[u8]> {
        match &self.0 {
            InnerArray::Owned(d) => d.metadata.as_ref().map(|b| b.as_slice()),
            InnerArray::Viewed(v) => v.flatbuffer().metadata().map(|m| m.bytes()),
        }
    }

    pub fn nbuffers(&self) -> usize {
        match &self.0 {
            InnerArray::Owned(o) => o.buffers.as_ref().map_or(0, |b| b.len()),
            InnerArray::Viewed(v) => v.nbuffers(),
        }
    }

    pub fn byte_buffer(&self, index: usize) -> Option<&ByteBuffer> {
        match &self.0 {
            InnerArray::Owned(d) => d.byte_buffer(index),
            InnerArray::Viewed(v) => v.buffer(index),
        }
    }

    pub fn byte_buffers(&self) -> impl Iterator<Item = ByteBuffer> + '_ {
        (0..self.nbuffers())
            .map(|i| self.byte_buffer(i).vortex_expect("missing declared buffer"))
            .cloned()
    }

    pub fn into_byte_buffer(self, index: usize) -> Option<ByteBuffer> {
        // NOTE(ngates): we can't really into_inner an Arc, so instead we clone the buffer out,
        //  but we still consume self by value such that the ref-count drops at the end of this
        //  function.
        match &self.0 {
            InnerArray::Owned(d) => d.byte_buffer(index).cloned(),
            InnerArray::Viewed(v) => v.buffer(index).cloned(),
        }
    }

    pub fn into_array_iterator(self) -> impl ArrayIterator {
        let dtype = self.dtype().clone();
        let iter = ChunkedArray::maybe_from(self.clone())
            .map(|chunked| ArrayChunkIterator::Chunked(chunked, 0))
            .unwrap_or_else(|| ArrayChunkIterator::Single(Some(self)));
        ArrayIteratorAdapter::new(dtype, iter)
    }

    pub fn into_array_stream(self) -> impl ArrayStream {
        ArrayStreamAdapter::new(
            self.dtype().clone(),
            futures_util::stream::iter(self.into_array_iterator()),
        )
    }

    /// Checks whether array is of a given encoding.
    pub fn is_encoding(&self, id: EncodingId) -> bool {
        self.encoding() == id
    }

    #[cfg(feature = "canonical_counter")]
    pub(crate) fn inc_canonical_counter(&self) {
        let prev = match &self.0 {
            InnerArray::Owned(o) => o
                .canonical_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            InnerArray::Viewed(v) => v
                .canonical_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        if prev >= 1 {
            log::warn!("Array::into_canonical called {} times on array", prev + 1,);
        }
        if prev >= 2 {
            let bt = backtrace::Backtrace::new();
            log::warn!("{:?}", bt);
        }
    }

    pub fn try_downcast_ref<E: Encoding>(&self) -> VortexResult<(&E::Array, &E)>
    where
        for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
    {
        let array_ref = <&E::Array>::try_from(self)?;
        let encoding = self
            .vtable()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        Ok((array_ref, encoding))
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = match &self.0 {
            InnerArray::Owned(_) => "",
            InnerArray::Viewed(_) => "$",
        };
        write!(
            f,
            "{}{}({}, len={})",
            prefix,
            self.encoding(),
            self.dtype(),
            self.len()
        )
    }
}

/// We define a single iterator that can handle both chunked and non-chunked arrays.
/// This avoids the need to create boxed static iterators for the two chunked and non-chunked cases.
enum ArrayChunkIterator {
    Single(Option<Array>),
    Chunked(ChunkedArray, usize),
}

impl Iterator for ArrayChunkIterator {
    type Item = VortexResult<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ArrayChunkIterator::Single(array) => array.take().map(Ok),
            ArrayChunkIterator::Chunked(chunked, idx) => (*idx < chunked.nchunks()).then(|| {
                let chunk = chunked.chunk(*idx);
                *idx += 1;
                chunk
            }),
        }
    }
}
