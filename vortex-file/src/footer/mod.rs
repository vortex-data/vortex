mod postscript;
mod segment;

use std::sync::Arc;

use flatbuffers::{FlatBufferBuilder, root};
use itertools::Itertools;
pub(crate) use postscript::*;
pub use segment::*;
use vortex_array::stats::StatsSet;
use vortex_array::{ArrayContext, ArrayRegistry};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};
use vortex_flatbuffers::{
    FlatBuffer, FlatBufferRoot, ReadFlatBuffer, WriteFlatBuffer, footer as fb,
};
use vortex_layout::{Layout, LayoutContext, LayoutRegistry};

/// Captures the layout information of a Vortex file.
#[derive(Debug, Clone)]
pub struct Footer {
    ctx: ArrayContext,
    layout_ctx: LayoutContext,
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
        ctx: ArrayContext,
        layout_ctx: LayoutContext,
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
            ctx,
            layout_ctx,
            layout: root_layout,
            segments,
            statistics,
        }
    }

    /// Read the [`Footer`] from a flatbuffer.
    pub fn read_flatbuffer(
        flatbuffer: FlatBuffer,
        dtype: DType,
        array_registry: &ArrayRegistry,
        layout_registry: &LayoutRegistry,
    ) -> VortexResult<Self> {
        let fb = root::<fb::Footer>(&flatbuffer)?;
        let fb_root_layout = fb
            .layout()
            .ok_or_else(|| vortex_err!("Footer missing root layout"))?;

        // Create a LayoutContext from the registry.
        let layout_encodings = fb.layout_encodings();
        let layout_ids = layout_encodings
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| encoding.id());
        let layout_ctx = layout_registry.new_context(layout_ids)?;

        // Create an ArrayContext from the registry.
        let array_encodings = fb.array_encodings();
        let array_ids = array_encodings
            .iter()
            .flat_map(|e| e.iter())
            .map(|encoding| encoding.id());
        let ctx = array_registry.new_context(array_ids)?;

        let root_encoding = layout_ctx
            .lookup_encoding(fb_root_layout.encoding())
            .ok_or_else(|| {
                vortex_err!(
                    "Footer root layout encoding {} not found",
                    fb_root_layout.encoding()
                )
            })?
            .clone();

        // SAFETY: We have validated the fb_root_layout at the beginning of this function
        let root_layout = unsafe {
            Layout::new_viewed_unchecked(
                "$".into(),
                root_encoding,
                dtype,
                flatbuffer.clone(),
                fb_root_layout._tab.loc(),
                layout_ctx.clone(),
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

        Ok(Self::new(
            ctx,
            layout_ctx,
            root_layout,
            segments,
            statistics,
        ))
    }

    /// Returns the array [`ArrayContext`] of the file.
    pub fn ctx(&self) -> &ArrayContext {
        &self.ctx
    }

    /// Returns the [`LayoutContext`] of the file.
    pub fn layout_ctx(&self) -> &LayoutContext {
        &self.layout_ctx
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

    /// Creates a [`FlatBufferRoot`] for the footer that can be used to serialize the footer
    /// into a flatbuffer.
    pub(crate) fn flatbuffer_writer<'a>(
        ctx: ArrayContext,
        layout: Layout,
        segments: Arc<[Segment]>,
        statistics: Option<Arc<[StatsSet]>>,
    ) -> impl WriteFlatBuffer<Target<'a> = fb::Footer<'a>> + FlatBufferRoot {
        FooterFlatBufferWriter {
            ctx,
            layout,
            segments,
            statistics,
        }
    }
}

pub(crate) struct FooterFlatBufferWriter {
    ctx: ArrayContext,
    layout: Layout,
    segments: Arc<[Segment]>,
    statistics: Option<Arc<[StatsSet]>>,
}

impl FlatBufferRoot for FooterFlatBufferWriter {}

impl WriteFlatBuffer for FooterFlatBufferWriter {
    type Target<'a> = fb::Footer<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> flatbuffers::WIPOffset<Self::Target<'fb>> {
        // Set up a layout context to capture the layouts used in the file.
        let layout_ctx = LayoutContext::empty();
        let layout = self.layout.write_flatbuffer(fbb, &layout_ctx);

        let segments = fbb.create_vector_from_iter(self.segments.iter().map(fb::Segment::from));
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
                fb::ArrayEncoding::create(fbb, &fb::ArrayEncodingArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let array_encodings = fbb.create_vector(array_encodings.as_slice());

        let layout_encodings = layout_ctx
            .encodings()
            .iter()
            .map(|e| {
                let id = fbb.create_string(e.id().as_ref());
                fb::LayoutEncoding::create(fbb, &fb::LayoutEncodingArgs { id: Some(id) })
            })
            .collect::<Vec<_>>();
        let layout_encodings = fbb.create_vector(layout_encodings.as_slice());

        fb::Footer::create(
            fbb,
            &fb::FooterArgs {
                layout: Some(layout),
                segments: Some(segments),
                statistics,
                array_encodings: Some(array_encodings),
                layout_encodings: Some(layout_encodings),
            },
        )
    }
}
