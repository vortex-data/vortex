use std::sync::Arc;

use vortex_flatbuffers::{footer2 as fb, FlatBufferRoot, WriteFlatBuffer};
use vortex_layout::LayoutData;

use crate::v2::footer::segment::Segment;

/// Captures the layout information of a Vortex file.
#[derive(Clone)]
pub(crate) struct FileLayout {
    pub(crate) root_layout: LayoutData,
    pub(crate) segments: Arc<[Segment]>,
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
