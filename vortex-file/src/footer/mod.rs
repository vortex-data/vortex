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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

pub(crate) use file_layout::*;
pub(crate) use file_statistics::*;
use flatbuffers::{root, root_unchecked};
use itertools::Itertools;
pub(crate) use postscript::*;
pub use segment::*;
use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::{ArrayContext, ArrayRegistry};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::scalar::ScalarValue;
use vortex_flatbuffers::{FlatBuffer, footer as fb};
use vortex_layout::{Layout, LayoutContext, LayoutRegistry};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    array_ctx: OnceLock<ArrayContext>,
    layout_ctx: OnceLock<LayoutContext>,
    root_layout: OnceLock<Layout>,
    segments: OnceLock<Arc<[SegmentSpec]>>,
    statistics: Option<FileStatistics>,

    flatbuffer: FlatBuffer,
    dtype: DType,
    array_registry: Arc<ArrayRegistry>,
    layout_registry: Arc<LayoutRegistry>,
    validated: Arc<AtomicBool>,
}

impl Footer {
    /// Read the [`Footer`] from a flatbuffer.
    pub(crate) fn from_flatbuffer(
        flatbuffer: FlatBuffer,
        dtype: DType,
        statistics: Option<FileStatistics>,
        array_registry: Arc<ArrayRegistry>,
        layout_registry: Arc<LayoutRegistry>,
    ) -> Self {
        Self {
            array_ctx: OnceLock::new(),
            layout_ctx: OnceLock::new(),
            root_layout: OnceLock::new(),
            segments: OnceLock::new(),
            statistics,
            flatbuffer,
            dtype,
            array_registry,
            layout_registry,
            validated: Arc::new(AtomicBool::new(false)),
        }
    }

    fn fb(&self) -> VortexResult<fb::FileLayout> {
        match self
            .validated
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(root::<fb::FileLayout>(&self.flatbuffer)?),
            Err(_) => Ok(unsafe { root_unchecked::<fb::FileLayout>(&self.flatbuffer) }),
        }
    }

    /// Returns the array [`ArrayContext`] of the file.
    pub fn ctx(&self) -> VortexResult<&ArrayContext> {
        self.array_ctx.get_or_try_init(|| {
            let array_specs = self.fb()?.array_specs();
            let array_ids = array_specs
                .iter()
                .flat_map(|e| e.iter())
                .map(|encoding| encoding.id());
            self.array_registry.new_context(array_ids)
        })
    }

    /// Returns the [`LayoutContext`] of the file.
    pub fn layout_ctx(&self) -> VortexResult<&LayoutContext> {
        self.layout_ctx.get_or_try_init(|| {
            let layout_specs = self.fb()?.layout_specs();
            let layout_ids = layout_specs
                .iter()
                .flat_map(|e| e.iter())
                .map(|encoding| encoding.id());
            self.layout_registry.new_context(layout_ids)
        })
    }

    /// Returns the root [`Layout`] of the file.
    pub fn layout(&self) -> VortexResult<&Layout> {
        self.root_layout.get_or_try_init(|| {
            let fb_root_layout = self
                .fb()?
                .layout()
                .ok_or_else(|| vortex_err!("Footer missing root layout"))?;

            let root_encoding = self
                .layout_ctx()?
                .lookup_encoding(fb_root_layout.encoding())
                .ok_or_else(|| {
                    vortex_err!(
                        "Footer root layout encoding {} not found",
                        fb_root_layout.encoding()
                    )
                })?
                .clone();

            // SAFETY: We have validated the fb_root_layout at the beginning of this function
            Ok(unsafe {
                Layout::new_viewed_unchecked(
                    "".into(),
                    root_encoding,
                    self.dtype.clone(),
                    self.flatbuffer.clone(),
                    fb_root_layout._tab.loc(),
                    self.layout_ctx()?.clone(),
                )
            })
        })
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> VortexResult<&Arc<[SegmentSpec]>> {
        self.segments.get_or_try_init(|| {
            let segments: Arc<[SegmentSpec]> = self
                .fb()?
                .segment_specs()
                .ok_or_else(|| vortex_err!("FileLayout missing segment specs"))?
                .iter()
                .map(SegmentSpec::try_from)
                .try_collect()?;

            // Note this assertion is `<=` since we allow zero-length segments
            if !segments.is_sorted_by_key(|segment| segment.offset) {
                vortex_bail!("Segment offsets are not ordered");
            }
            Ok(segments)
        })
    }

    /// Returns the statistics of the file.
    pub fn statistics(&self) -> Option<&Arc<[StatsSet]>> {
        self.statistics.as_ref().map(|s| &s.0)
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> VortexResult<u64> {
        Ok(self.layout()?.row_count())
    }

    pub fn nbytes(&self) -> usize {
        let stats_size = self.statistics().iter().map(|v| v.len()).sum::<usize>()
            * (size_of::<Stat>() + size_of::<Precision<ScalarValue>>());

        size_of::<DType>() + stats_size + self.flatbuffer.len()
    }
}
