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
use flatbuffers::root;
use itertools::Itertools;
pub(crate) use postscript::*;
pub use segment::*;
use vortex_array::ArrayContext;
use vortex_array::ArrayRegistry;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, footer as fb};
use vortex_io::VortexWrite;
use vortex_layout::{LayoutContext, LayoutRef, LayoutRegistry, layout_from_flatbuffer};

use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    root_layout: LayoutRef,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
}

struct PositionAndLength {
    offset: u64,
    length: u32,
}

impl PositionAndLength {
    pub fn into_postscript_segment(
        self,
        postscript_offset: u64,
    ) -> VortexResult<PostscriptSegment> {
        let offset = i64::try_from(self.offset).map_err(|_| vortex_err!("offset too big"))?;
        let postscript_offset =
            i64::try_from(postscript_offset).map_err(|_| vortex_err!("file too big"))?;
        Ok(PostscriptSegment {
            offset: offset - postscript_offset,
            length: self.length,
            alignment: FlatBuffer::alignment(),
        })
    }
}

async fn write_flatbuffer<W: VortexWrite, F: FlatBufferRoot + WriteFlatBuffer>(
    write: &mut futures::io::Cursor<W>,
    flatbuffer: &F,
) -> VortexResult<PositionAndLength> {
    let offset = write.position();

    write.write_all(flatbuffer.write_flatbuffer_bytes()).await?;

    let length = write.position() - offset;
    let length =
        u32::try_from(length).map_err(|_| vortex_err!("segment length exceeds maximum u32"))?;

    Ok(PositionAndLength { offset, length })
}

impl Footer {
    pub fn from_parts(
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

    pub(crate) async fn write<W: VortexWrite>(
        self,
        mut write: futures::io::Cursor<W>,
        array_ctx: ArrayContext,
        dtype_segment: Option<PostscriptSegment>,
    ) -> VortexResult<W> {
        let layout_ctx = LayoutContext::empty();
        let layout_segment =
            write_flatbuffer(&mut write, &self.root_layout.flatbuffer_writer(&layout_ctx)).await?;

        let statistics_segment = if let Some(ref statistics) = self.statistics {
            Some(write_flatbuffer(&mut write, statistics).await?)
        } else {
            None
        };

        let footer_flatbuffer_writer = FooterFlatBufferWriter {
            ctx: array_ctx,
            layout_ctx,
            segment_specs: self.segments,
        };
        let footer_segment = write_flatbuffer(&mut write, &footer_flatbuffer_writer).await?;

        let postscript_offset = write.position();

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            layout: layout_segment.into_postscript_segment(postscript_offset)?,
            statistics: statistics_segment
                .map(|x| x.into_postscript_segment(postscript_offset))
                .transpose()?,
            footer: footer_segment.into_postscript_segment(postscript_offset)?,
        };
        let postscript_buffer = postscript.write_flatbuffer_bytes();
        if postscript_buffer.len() > MAX_FOOTER_SIZE as usize {
            vortex_bail!(
                "Postscript is too large ({} bytes); max postscript size is {}",
                postscript_buffer.len(),
                MAX_FOOTER_SIZE
            );
        }
        let postscript_len = u16::try_from(postscript_buffer.len())
            .vortex_expect("Postscript already verified to fit into u16");
        write.write_all(postscript_buffer).await?;

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        write.write_all(eof).await?;

        write.flush().await?;

        Ok(write.into_inner())
    }
    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_flatbuffer(
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
