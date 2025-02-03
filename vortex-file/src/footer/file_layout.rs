use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_flatbuffers::{footer as fb, FlatBufferRoot, WriteFlatBuffer};
use vortex_layout::Layout;

use crate::footer::segment::Segment;

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct FileLayout {
    root_layout: Layout,
    segments: Arc<[Segment]>,
}

impl FileLayout {
    /// Create a new `FileLayout` from the root layout and segments.
    ///
    /// ## Panics
    ///
    /// Panics if the segments are not ordered by byte offset.
    pub fn new(root_layout: Layout, segments: Arc<[Segment]>) -> Self {
        // Note this assertion is `<=` since we allow zero-length segments
        assert!(segments
            .iter()
            .tuple_windows()
            .all(|(a, b)| a.offset <= b.offset));
        Self {
            root_layout,
            segments,
        }
    }

    /// Returns the root [`Layout`] of the file.
    pub fn root_layout(&self) -> &Layout {
        &self.root_layout
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> &Arc<[Segment]> {
        &self.segments
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        self.root_layout.dtype()
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.root_layout.row_count()
    }
}

impl FlatBufferRoot for FileLayout {}

impl WriteFlatBuffer for FileLayout {
    type Target<'a> = fb::FileLayout<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let root_layout = self.root_layout.write_flatbuffer(fbb);
        let segments = fbb.create_vector_from_iter(self.segments.iter().map(fb::Segment::from));
        fb::FileLayout::create(
            fbb,
            &fb::FileLayoutArgs {
                root_layout: Some(root_layout),
                segments: Some(segments),
            },
        )
    }
}
