use flatbuffers::{FlatBufferBuilder, WIPOffset};
use vortex_flatbuffers::{footer as fb, WriteFlatBuffer};

use crate::layouts::write::layouts::LayoutSpec;

#[derive(Debug)]
pub struct Footer {
    layout: LayoutSpec,
    row_count: u64,
}

impl Footer {
    pub fn new(layout: LayoutSpec, row_count: u64) -> Self {
        Self { layout, row_count }
    }
}

impl WriteFlatBuffer for Footer {
    type Target<'a> = fb::Footer<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let layout_offset = self.layout.write_flatbuffer(fbb);
        fb::Footer::create(
            fbb,
            &fb::FooterArgs {
                layout: Some(layout_offset),
                row_count: self.row_count,
            },
        )
    }
}

#[derive(Debug)]
pub struct Postscript {
    schema_offset: u64,
    footer_offset: u64,
}

impl Postscript {
    pub fn new(schema_offset: u64, footer_offset: u64) -> Self {
        Self {
            schema_offset,
            footer_offset,
        }
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
                footer_offset: self.footer_offset,
            },
        )
    }
}
