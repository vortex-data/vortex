use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::future::ready;
use std::sync::{Arc, RwLock};

use itertools::Itertools;
use owned::OwnedArrayData;
use viewed::ViewedArrayData;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{
    BoolEncoding, ExtensionEncoding, NullEncoding, PrimitiveEncoding, StructEncoding,
    VarBinEncoding, VarBinViewEncoding,
};
use crate::compute::scalar_at;
use crate::encoding::{Encoding, EncodingId, EncodingRef, EncodingVTable};
use crate::iter::{ArrayIterator, ArrayIteratorAdapter};
use crate::stats::{ArrayStatistics, Stat, Statistics, StatsSet};
use crate::stream::{ArrayStream, ArrayStreamAdapter};
use crate::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use crate::{
    ArrayChildrenIterator, ArrayDType, ArrayLen, ArrayMetadata, ContextRef, NamedChildrenCollector,
    TryDeserializeArrayMetadata,
};

mod owned;
mod viewed;

/// A central type for all Vortex arrays, which are known length sequences of typed and possibly compressed data.
///
/// This is the main entrypoint for working with in-memory Vortex data, and dispatches work over the underlying encoding or memory representations.
#[derive(Debug, Clone)]
pub struct ArrayData(InnerArrayData);

#[derive(Debug, Clone)]
enum InnerArrayData {
    /// Owned [`ArrayData`] with serialized metadata, backed by heap-allocated memory.
    Owned(OwnedArrayData),
    /// Zero-copy view over flatbuffer-encoded [`ArrayData`] data, created without eager serialization.
    Viewed(ViewedArrayData),
}

impl From<OwnedArrayData> for ArrayData {
    fn from(data: OwnedArrayData) -> Self {
        ArrayData(InnerArrayData::Owned(data))
    }
}

impl From<ViewedArrayData> for ArrayData {
    fn from(data: ViewedArrayData) -> Self {
        ArrayData(InnerArrayData::Viewed(data))
    }
}

impl ArrayData {
    pub fn try_new_owned(
        encoding: EncodingRef,
        dtype: DType,
        len: usize,
        metadata: Arc<dyn ArrayMetadata>,
        buffers: Arc<[ByteBuffer]>,
        children: Arc<[ArrayData]>,
        statistics: StatsSet,
    ) -> VortexResult<Self> {
        Self::try_new(InnerArrayData::Owned(OwnedArrayData {
            encoding,
            dtype,
            len,
            metadata,
            buffers,
            children,
            stats_set: Arc::new(RwLock::new(statistics)),
            #[cfg(feature = "canonical_counter")]
            canonical_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }))
    }

    pub fn try_new_viewed<F>(
        ctx: ContextRef,
        dtype: DType,
        len: usize,
        // TODO(ngates): use ConstByteBuffer
        flatbuffer: ByteBuffer,
        flatbuffer_init: F,
        buffers: Vec<ByteBuffer>,
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

        // Parse the array metadata
        let metadata = encoding.load_metadata(array.metadata().map(|v| v.bytes()))?;

        let view = ViewedArrayData {
            encoding,
            dtype,
            len,
            metadata,
            flatbuffer,
            flatbuffer_loc,
            buffers: buffers.into(),
            ctx,
            #[cfg(feature = "canonical_counter")]
            canonical_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };

        Self::try_new(InnerArrayData::Viewed(view))
    }

    /// Shared constructor that performs common array validation.
    fn try_new(inner: InnerArrayData) -> VortexResult<Self> {
        let array = ArrayData(inner);

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
            array.encoding().id(),
            array.dtype()
        );

        // TODO(ngates): we should run something like encoding.validate(array) so the encoding
        //  can check the number of children, number of buffers, and metadata values etc.

        Ok(array)
    }

    /// Return the array's encoding
    pub fn encoding(&self) -> EncodingRef {
        match &self.0 {
            InnerArrayData::Owned(d) => d.encoding,
            InnerArrayData::Viewed(v) => v.encoding,
        }
    }

    /// Returns the number of logical elements in the array.
    #[allow(clippy::same_name_method)]
    pub fn len(&self) -> usize {
        match &self.0 {
            InnerArrayData::Owned(d) => d.len,
            InnerArrayData::Viewed(v) => v.len,
        }
    }

    /// Check whether the array has any data
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the array is of a canonical encoding.
    pub fn is_canonical(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(StructEncoding.id())
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
            .get_as::<bool>(Stat::IsConstant)
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
            InnerArrayData::Owned(d) => d.child(idx, dtype, len).cloned(),
            InnerArrayData::Viewed(v) => v
                .child(idx, dtype, len)
                .map(|view| ArrayData(InnerArrayData::Viewed(view))),
        }
    }

    /// Returns a Vec of Arrays with all the array's child arrays.
    pub fn children(&self) -> Vec<ArrayData> {
        match &self.0 {
            InnerArrayData::Owned(d) => d.children().to_vec(),
            InnerArrayData::Viewed(v) => v.children(),
        }
    }

    /// Returns a Vec of Arrays with all the array's child arrays.
    pub fn named_children(&self) -> Vec<(String, ArrayData)> {
        let mut collector = NamedChildrenCollector::default();
        self.encoding()
            .accept(&self.clone(), &mut collector)
            .vortex_expect("Failed to get children");
        collector.children()
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

    pub fn array_metadata(&self) -> &dyn ArrayMetadata {
        match &self.0 {
            InnerArrayData::Owned(d) => &*d.metadata,
            InnerArrayData::Viewed(v) => &*v.metadata,
        }
    }

    pub fn metadata<M: ArrayMetadata + Clone + for<'m> TryDeserializeArrayMetadata<'m>>(
        &self,
    ) -> VortexResult<&M> {
        match &self.0 {
            InnerArrayData::Owned(d) => &d.metadata,
            InnerArrayData::Viewed(v) => &v.metadata,
        }
        .as_any()
        .downcast_ref::<M>()
        .ok_or_else(|| {
            vortex_err!(
                "Failed to downcast metadata to {}",
                std::any::type_name::<M>()
            )
        })
    }

    /// Get back the (possibly owned) metadata for the array.
    ///
    /// View arrays will return a reference to their bytes, while heap-backed arrays
    /// must first serialize their metadata, returning an owned byte array to the caller.
    pub fn metadata_bytes(&self) -> VortexResult<Cow<[u8]>> {
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
                    .metadata_bytes()
                    .ok_or_else(|| vortex_err!("things"))
                    .map(Cow::Borrowed)
            }
        }
    }

    pub fn nbuffers(&self) -> usize {
        match &self.0 {
            InnerArrayData::Owned(o) => o.buffers.len(),
            InnerArrayData::Viewed(v) => v.nbuffers(),
        }
    }

    pub fn byte_buffer(&self, index: usize) -> Option<&ByteBuffer> {
        match &self.0 {
            InnerArrayData::Owned(d) => d.byte_buffer(index),
            InnerArrayData::Viewed(v) => v.buffer(index),
        }
    }

    pub fn byte_buffers(&self) -> impl Iterator<Item = ByteBuffer> + '_ {
        (0..self.nbuffers())
            .map(|i| self.byte_buffer(i).vortex_expect("missing declared buffer"))
            .cloned()
    }

    pub fn into_byte_buffer(self, index: usize) -> Option<ByteBuffer> {
        match self.0 {
            InnerArrayData::Owned(d) => d.into_byte_buffer(index),
            InnerArrayData::Viewed(v) => v.buffer(index).cloned(),
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

    #[cfg(feature = "canonical_counter")]
    pub(crate) fn inc_canonical_counter(&self) {
        let prev = match &self.0 {
            InnerArrayData::Owned(o) => o
                .canonical_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            InnerArrayData::Viewed(v) => v
                .canonical_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        if prev >= 1 {
            log::warn!(
                "ArrayData::into_canonical called {} times on array",
                prev + 1,
            );
        }
        if prev >= 2 {
            let bt = backtrace::Backtrace::new();
            log::warn!("{:?}", bt);
        }
    }

    pub fn downcast_array_ref<E: Encoding>(self: &ArrayData) -> VortexResult<(&E::Array, &E)>
    where
        for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
    {
        let array_ref = <&E::Array>::try_from(self)?;
        let encoding = self
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        Ok((array_ref, encoding))
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

impl<T: AsRef<ArrayData>> ArrayDType for T {
    fn dtype(&self) -> &DType {
        match &self.as_ref().0 {
            InnerArrayData::Owned(d) => &d.dtype,
            InnerArrayData::Viewed(v) => &v.dtype,
        }
    }
}

impl<T: AsRef<ArrayData>> ArrayLen for T {
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }
}

impl<A: AsRef<ArrayData>> ArrayValidity for A {
    /// Return whether the element at the given index is valid (true) or null (false).
    fn is_valid(&self, index: usize) -> bool {
        ValidityVTable::<ArrayData>::is_valid(self.as_ref().encoding(), self.as_ref(), index)
    }

    /// Return the logical validity of the array.
    fn logical_validity(&self) -> LogicalValidity {
        ValidityVTable::<ArrayData>::logical_validity(self.as_ref().encoding(), self.as_ref())
    }
}

impl<T: AsRef<ArrayData>> ArrayStatistics for T {
    fn statistics(&self) -> &(dyn Statistics + '_) {
        match &self.as_ref().0 {
            InnerArrayData::Owned(d) => d,
            InnerArrayData::Viewed(v) => v,
        }
    }

    // FIXME(ngates): this is really slow...
    fn inherit_statistics(&self, parent: &dyn Statistics) {
        let stats = self.statistics();
        // The to_set call performs a slow clone of the stats
        for (stat, scalar) in parent.to_set() {
            stats.set(stat, scalar);
        }
    }
}
