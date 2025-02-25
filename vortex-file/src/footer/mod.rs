mod postscript;
mod segment;

use std::sync::Arc;

use flatbuffers::root;
use itertools::Itertools;
pub(crate) use postscript::*;
pub use segment::*;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
use vortex_flatbuffers::{
    FlatBuffer, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb,
};
use vortex_layout::{Layout, LayoutContextRef, LayoutId};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    layout: Layout,
    segments: Arc<[Segment]>,
    statistics: Option<Arc<[StatsSet]>>,
}

impl Footer {
    /// Create a new `Footer` from the root layout and segments.
    ///
    /// ## Panics
    ///
    /// Panics if the segments are not ordered by byte offset.
    pub fn new(
        root_layout: Layout,
        segments: Arc<[Segment]>,
        statistics: Option<Arc<[StatsSet]>>,
    ) -> Self {
        // Note this assertion is `<=` since we allow zero-length segments
        assert!(
            segments
                .iter()
                .tuple_windows()
                .all(|(a, b)| a.offset <= b.offset)
        );
        Self {
            layout: root_layout,
            segments,
            statistics,
        }
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub fn read_flatbuffer(
        flatbuffer: FlatBuffer,
        ctx: &LayoutContextRef,
        dtype: DType,
    ) -> VortexResult<Self> {
        let fb = root::<fb::Footer>(&flatbuffer)?;
        let fb_root_layout = fb
            .layout()
            .ok_or_else(|| vortex_err!("Footer missing root layout"))?;

        let root_encoding = ctx
            .lookup_layout(LayoutId(fb_root_layout.encoding()))
            .ok_or_else(|| {
                vortex_err!(
                    "Footer root layout encoding {} not found",
                    fb_root_layout.encoding()
                )
            })?;

        // SAFETY: We have validated the fb_root_layout at the beginning of this function
        let root_layout = unsafe {
            Layout::new_viewed_unchecked(
                "$".into(),
                root_encoding,
                dtype,
                flatbuffer.clone(),
                fb_root_layout._tab.loc(),
                ctx.clone(),
            )
        };

        let fb_segments = fb
            .segments()
            .ok_or_else(|| vortex_err!("Footer missing segments"))?;
        let segments = fb_segments.iter().map(Segment::try_from).try_collect()?;

        let statistics = fb
            .statistics()
            .map(|s| {
                s.iter()
                    .map(|s| StatsSet::read_flatbuffer(&s))
                    .try_collect()
            })
            .transpose()?;

        Ok(Self::new(root_layout, segments, statistics))
    }

    /// Returns the root [`Layout`] of the file.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the segment map of the file.
    pub fn segment_map(&self) -> &Arc<[Segment]> {
        &self.segments
    }

    /// Returns the statistics of the file.
    pub fn statistics(&self) -> Option<&Arc<[StatsSet]>> {
        self.statistics.as_ref()
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }
}

impl FlatBufferRoot for Footer {}

impl WriteFlatBuffer for Footer {
    type Target<'a> = fb::Footer<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut flatbuffers::FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        let layout = self.layout.write_flatbuffer(fbb);
        let segments = fbb.create_vector_from_iter(self.segments.iter().map(fb::Segment::from));
        let statistics = self
            .statistics()
            .map(|stats| stats.iter().map(|s| s.write_flatbuffer(fbb)).collect_vec());
        let statistics = statistics.map(|s| fbb.create_vector(s.as_slice()));

        fb::Footer::create(
            fbb,
            &fb::FooterArgs {
                layout: Some(layout),
                segments: Some(segments),
                statistics,
            },
        )
    }
}
