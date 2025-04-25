//! This module defines the file layout component of the Vortex file footer.
//!
//! The file layout describes the structure of the data in the file, including:
//! - The root layout of the file
//! - Specifications for all segments in the file
//! - Specifications for array and layout encodings used in the file
use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_array::ArrayContext;
use vortex_flatbuffers::{FlatBufferRoot, WriteFlatBuffer, footer as fb};
use vortex_layout::LayoutContext;

use crate::footer::segment::SegmentSpec;

/// A writer for serializing a file layout to a FlatBuffer.
///
/// This struct is used to write the layout component of a Vortex file footer,
/// which describes the structure of the data in the file.
pub(crate) struct FooterFlatBufferWriter {
    /// The array context containing encodings used in the file.
    pub(crate) ctx: ArrayContext,
    /// The layout context containing the layouts used in the file.
    pub(crate) layout_ctx: LayoutContext,
    /// Specifications for all segments in the file.
    pub(crate) segment_specs: Arc<[SegmentSpec]>,
}

impl FlatBufferRoot for FooterFlatBufferWriter {}

impl WriteFlatBuffer for FooterFlatBufferWriter {
    type Target<'a> = fb::Footer<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let segment_specs =
            fbb.create_vector_from_iter(self.segment_specs.iter().map(fb::SegmentSpec::from));

        let array_specs = self
            .ctx
            .encodings()
            .iter()
            .map(|e| {
                let id = fbb.create_string(e.id().as_ref());
                fb::ArraySpec::create(fbb, &fb::ArraySpecArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let array_specs = fbb.create_vector(array_specs.as_slice());

        let layout_specs = self
            .layout_ctx
            .encodings()
            .iter()
            .map(|e| {
                let id = fbb.create_string(e.id().as_ref());
                fb::LayoutSpec::create(fbb, &fb::LayoutSpecArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let layout_specs = fbb.create_vector(layout_specs.as_slice());

        fb::Footer::create(
            fbb,
            &fb::FooterArgs {
                segment_specs: Some(segment_specs),
                array_specs: Some(array_specs),
                layout_specs: Some(layout_specs),
                compression_specs: None,
                encryption_specs: None,
            },
        )
    }
}
