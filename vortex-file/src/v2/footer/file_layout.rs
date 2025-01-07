use vortex_dtype::DType;
use vortex_flatbuffers::{footer2 as fb, FlatBufferRoot, WriteFlatBuffer};
use vortex_layout::LayoutData;

use crate::v2::footer::segment::Segment;

/// Captures the layout information of a Vortex file.
#[derive(Clone, Debug)]
pub struct FileLayout {
    pub(crate) root_layout: LayoutData,
    pub(crate) segments: Vec<Segment>,
}

impl FileLayout {
    /// The [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        &self.root_layout.dtype()
    }

    /// The row count of the file.
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

        let segments = self
            .segments
            .iter()
            .map(|segment| segment.write_flatbuffer(fbb))
            .collect::<Vec<_>>();
        let segments = fbb.create_vector(&segments);

        fb::FileLayout::create(
            fbb,
            &fb::FileLayoutArgs {
                root_layout: Some(root_layout),
                segments: Some(segments),
            },
        )
    }
}
