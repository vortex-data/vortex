use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_error::{vortex_bail, VortexResult};
use vortex_flatbuffers::{footer as fb, WriteFlatBuffer};

#[derive(Debug)]
pub struct Postscript {
    schema_offset: u64,
    layout_offset: u64,
}

impl Postscript {
    pub fn try_new(schema_offset: u64, layout_offset: u64) -> VortexResult<Self> {
        if layout_offset < schema_offset {
            vortex_bail!(
                "layout_offset ({}) must be greater than or equal to schema_offset ({})",
                layout_offset,
                schema_offset
            );
        }
        Ok(Self {
            schema_offset,
            layout_offset,
        })
    }
}

impl WriteFlatBuffer for Postscript {
    type Target<'a> = fb::Postscript<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        fb::Postscript::create(
            fbb,
            &fb::PostscriptArgs {
                schema_offset: self.schema_offset,
                layout_offset: self.layout_offset,
            },
        )
    }
}
