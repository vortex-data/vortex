use std::sync::{Arc, OnceLock};

use vortex_array::DeserializeMetadata;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::{LayoutReaderRef, VTable};

pub trait Layout {
    fn new_reader(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn ReaderChildren,
    ) -> VortexResult<LayoutReaderRef>;
}

pub type LayoutRef = Arc<dyn Layout>;

#[repr(transparent)]
pub struct LayoutAdapter<V: VTable>(V::Layout);

impl<V: VTable> Layout for LayoutAdapter<V> {
    fn new_reader(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn ReaderChildren,
    ) -> VortexResult<LayoutReaderRef> {
        let metadata = <V::Metadata as DeserializeMetadata>::deserialize(metadata)?;
        let reader =
            V::reader_from_parts(&self.0, dtype, row_count, &metadata, segment_ids, children)?;
        assert_eq!(
            reader.row_count(),
            row_count,
            "LayoutReader row count mismatch after building"
        );
        Ok(reader.to_layout_reader())
    }
}

pub trait ReaderChildren {
    /// Returns the nth child of the layout with the given dtype and length.
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<LayoutRef>;

    /// The number of children.
    fn len(&self) -> usize;

    /// Returns true if there are no children.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A utility for providing cached lazy access to children of a layout reader.
pub struct LazyReaderChildren {
    children: Box<dyn ReaderChildren>,
    // TODO(ngates): change this data structure based on the number of children..
    cache: Arc<[OnceLock<LayoutReaderRef>]>,
}
