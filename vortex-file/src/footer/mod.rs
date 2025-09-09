// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the footer of a Vortex file, which contains metadata about the file's contents.
//!
//! The footer includes:
//! - The file's layout, which describes how the data is organized
//! - Statistics about the data, which can be used for query optimization
//! - Segment map, which describe the physical location of data in the file
//!
//! The footer is located at the end of the file and is used to interpret the file's contents.
mod file_layout;
mod file_statistics;
mod postscript;
mod segment;

use std::sync::Arc;

pub(crate) use file_layout::*;
pub(crate) use file_statistics::*;
use flatbuffers::{FlatBufferBuilder, root};
use itertools::Itertools;
pub(crate) use postscript::*;
pub use segment::*;
use vortex_array::stats::StatsSet;
use vortex_array::{ArrayContext, ArrayRegistry};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, ReadFlatBuffer, WriteFlatBuffer, footer as fb};
use vortex_layout::{
    LayoutContext, LayoutRef, LayoutRegistry, layout_from_fb_layout, layout_from_flatbuffer,
};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
}

impl Footer {
    /// Deserialize a Footer from its FlatBuffer-serialized form.
    // We cannot impl ReadFlatBuffer because reading a Layout *requires* an owned bytes and the
    // ReadFlatBuffer interface would only provide an `fb::FullFooter`.
    pub fn from_flatbuffer(
        bytes: FlatBuffer,
        dtype: DType,
        layout_ctx: &LayoutContext,
        array_ctx: &ArrayContext,
    ) -> VortexResult<Self> {
        let full_footer = root::<fb::FullFooter>(bytes.as_ref())?;
        let root_layout = layout_from_fb_layout(
            bytes.clone(),
            full_footer
                .layout()
                .ok_or_else(|| vortex_err!("layout missing"))?,
            &dtype,
            layout_ctx,
            array_ctx,
        )?;
        let footer = full_footer
            .footer()
            .ok_or_else(|| vortex_err!("footer missing"))?;
        let segments = footer
            .segment_specs()
            .unwrap_or_default()
            .iter()
            .map(SegmentSpec::try_from)
            .try_collect()?;
        let statistics = full_footer
            .statistics()
            .as_ref()
            .map(FileStatistics::read_flatbuffer)
            .transpose()?;
        Ok(Footer {
            root_layout,
            segments,
            statistics,
        })
    }

    /// Serialize a Footer to FlatBuffer bytes.
    ///
    /// To later reconstruct a Footer with [Self::from_flatbuffer], you will also need the root
    /// [DType], which can likewise be serialized as a FlatBuffer.
    ///
    /// See also: [Self::write_flatbuffer].
    pub fn write_flatbuffer_bytes(
        &self,
        layout_ctx: &LayoutContext,
        array_ctx: &ArrayContext,
    ) -> FlatBuffer {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = self.write_flatbuffer(&mut fbb, layout_ctx, array_ctx);
        fbb.finish_minimal(root_offset);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        FlatBuffer::align_from(ByteBuffer::from(vec).slice(start..end))
    }

    /// Serialize a Footer to its FlatBuffer-serialized form.
    // We cannot implement WriteFlatBuffer because that trait requires us to be able to write into a
    // FlatBufferBuilder with *any* possible lifetime. We can only write into a builder with a
    // lifetime shorter than ours. Rust does not allow us to add this constraint to our
    // implementation.
    pub fn write_flatbuffer<'fb, 'a: 'fb>(
        &'a self,
        fbb: &mut FlatBufferBuilder<'fb>,
        layout_ctx: &'fb LayoutContext,
        array_ctx: &ArrayContext,
    ) -> flatbuffers::WIPOffset<fb::FullFooter<'fb>> {
        // The order inentionally mirrors teh order in vortex-file/src/writer.rs for pleasing symmetry.
        let layout = self
            .root_layout
            .flatbuffer_writer(layout_ctx)
            .write_flatbuffer(fbb);
        let file_statistics = self
            .statistics
            .as_ref()
            .map(|statistics| statistics.write_flatbuffer(fbb));
        let footer = FooterFlatBufferWriter {
            ctx: array_ctx.clone(),
            layout_ctx: layout_ctx.clone(),
            segment_specs: self.segments.clone(),
        }
        .write_flatbuffer(fbb);

        fb::FullFooter::create(
            fbb,
            &fb::FullFooterArgs {
                footer: Some(footer),
                layout: Some(layout),
                statistics: file_statistics,
            },
        )
    }
}

impl Footer {
    pub(crate) fn new(
        root_layout: LayoutRef,
        segments: Arc<[SegmentSpec]>,
        statistics: Option<FileStatistics>,
    ) -> Self {
        Self {
            root_layout,
            segments,
            statistics,
        }
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_parts(
        footer_bytes: FlatBuffer,
        layout_bytes: FlatBuffer,
        dtype: DType,
        statistics: Option<FileStatistics>,
        array_registry: &ArrayRegistry,
        layout_registry: &LayoutRegistry,
    ) -> VortexResult<Self> {
        let fb_footer = root::<fb::Footer>(&footer_bytes)?;

        // Create a LayoutContext from the registry.
        let layout_specs = fb_footer.layout_specs();
        let layout_ids = layout_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| encoding.id());
        let layout_ctx = layout_registry.new_context(layout_ids)?;

        // Create an ArrayContext from the registry.
        let array_specs = fb_footer.array_specs();
        let array_ids = array_specs
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| encoding.id());
        let array_ctx = array_registry.new_context(array_ids)?;

        let root_layout = layout_from_flatbuffer(layout_bytes, &dtype, &layout_ctx, &array_ctx)?;

        let segments: Arc<[SegmentSpec]> = fb_footer
            .segment_specs()
            .ok_or_else(|| vortex_err!("FileLayout missing segment specs"))?
            .iter()
            .map(SegmentSpec::try_from)
            .try_collect()?;

        // Note this assertion is `<=` since we allow zero-length segments
        if !segments.is_sorted_by_key(|segment| segment.offset) {
            vortex_bail!("Segment offsets are not ordered");
        }

        Ok(Self {
            root_layout,
            segments,
            statistics,
        })
    }

    /// Returns the root [`LayoutRef`] of the file.
    pub fn layout(&self) -> &LayoutRef {
        &self.root_layout
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> &Arc<[SegmentSpec]> {
        &self.segments
    }

    /// Returns the statistics of the file.
    pub fn statistics(&self) -> Option<&Arc<[StatsSet]>> {
        self.statistics.as_ref().map(|s| &s.0)
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
