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
use vortex_array::stats::StatsSet;
use vortex_array::{ArrayContext, ArrayRegistry};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, footer as fb, layout as fbl};
use vortex_layout::{Layout, LayoutContext, LayoutRegistry};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    array_ctx: ArrayContext,
    layout_ctx: LayoutContext,
    root_layout: Layout,
    segments: Arc<[SegmentSpec]>,
    statistics: Option<FileStatistics>,
}

impl Footer {
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
        let fb_layout = root::<fbl::Layout>(&layout_bytes)?;

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

        let root_encoding = layout_ctx
            .lookup_encoding(fb_layout.encoding())
            .ok_or_else(|| {
                vortex_err!(
                    "Footer root layout encoding {} not found",
                    fb_layout.encoding()
                )
            })?
            .clone();

        // SAFETY: We have validated the fb_root_layout at the beginning of this function
        let root_layout = unsafe {
            Layout::new_viewed_unchecked(
                "".into(),
                root_encoding,
                dtype,
                layout_bytes.clone(),
                fb_layout._tab.loc(),
                layout_ctx.clone(),
            )
        };

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
            array_ctx,
            layout_ctx,
            root_layout,
            segments,
            statistics,
        })
    }

    /// Returns the array [`ArrayContext`] of the file.
    pub fn ctx(&self) -> &ArrayContext {
        &self.array_ctx
    }

    /// Returns the [`LayoutContext`] of the file.
    pub fn layout_ctx(&self) -> &LayoutContext {
        &self.layout_ctx
    }

    /// Returns the root [`Layout`] of the file.
    pub fn layout(&self) -> &Layout {
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
