use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_flatbuffers::{FlatBufferRoot, WriteFlatBuffer, footer as fb};
use vortex_layout::LayoutContext;

pub struct RegistryFlatBufferWriter {}

impl FlatBufferRoot for RegistryFlatBufferWriter {}

impl WriteFlatBuffer for RegistryFlatBufferWriter {
    type Target<'a> = fb::Registry<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        // Set up a layout context to capture the layouts used in the file.
        let layout_ctx = LayoutContext::empty();
        let layout = self.layout.write_flatbuffer(fbb, &layout_ctx);

        let segments = fbb.create_vector_from_iter(self.segments.iter().map(fb::SegmentSpec::from));
        let statistics = self
            .statistics
            .as_ref()
            .map(|stats| stats.iter().map(|s| s.write_flatbuffer(fbb)).collect_vec());
        let statistics = statistics.map(|s| fbb.create_vector(s.as_slice()));

        let array_encodings = self
            .ctx
            .encodings()
            .iter()
            .map(|e| {
                let id = fbb.create_string(e.id().as_ref());
                fb::ArraySpec::create(fbb, &fb::ArraySpecArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let array_encodings = fbb.create_vector(array_encodings.as_slice());

        let layout_encodings = layout_ctx
            .encodings()
            .iter()
            .map(|e| {
                let id = fbb.create_string(e.id().as_ref());
                fb::LayoutSpec::create(fbb, &fb::LayoutSpecArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let layout_encodings = fbb.create_vector(layout_encodings.as_slice());

        fb::Registry::create(
            fbb,
            &fb::RegistryArgs {
                array_specs: None,
                layout_specs: None,
                segment_specs: None,
                compression_specs: None,
                encryption_specs: None,
            },
        )
    }
}
