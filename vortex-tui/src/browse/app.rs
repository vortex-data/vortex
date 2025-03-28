use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use ratatui::widgets::ListState;
use vortex::buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex::file::{Footer, SegmentSpec, VortexOpenOptions};
use vortex::io::TokioFile;
use vortex::stats::stats_from_bitset_bytes;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::stats::StatsLayout;
use vortex_layout::layouts::stats::stats_table::StatsTable;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};
use vortex_layout::{
    CHUNKED_LAYOUT_ID, FLAT_LAYOUT_ID, Layout, LayoutVTable, LayoutVTableRef, STATS_LAYOUT_ID,
    STRUCT_LAYOUT_ID,
};

#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub enum Tab {
    /// The layout tree browser.
    #[default]
    Layout,
    /// The encoding tree viewer
    Encodings,
    // TODO(aduffy): SQL query page powered by DF
    // Query,
}

/// A pointer into the `Layout` hierarchy that can be advanced.
///
/// The pointer wraps an InitialRead.
pub struct LayoutCursor {
    path: Vec<usize>,
    footer: Footer,
    layout: Layout,
    segment_map: Arc<[SegmentSpec]>,
}

impl LayoutCursor {
    pub fn new(footer: Footer) -> Self {
        Self {
            path: Vec::new(),
            layout: footer.layout().clone(),
            segment_map: Arc::clone(footer.segment_map()),
            footer,
        }
    }

    pub fn new_with_path(footer: Footer, path: Vec<usize>) -> Self {
        let mut layout = footer.layout().clone();
        let mut dtype = footer.dtype().clone();
        // Traverse the layout tree at each element of the path.
        for component in path.iter().copied() {
            // Find the DType of the child based on the DType of the current node.
            // TODO(ngates): add visitor pattern to layout
            dtype = if layout.id() == CHUNKED_LAYOUT_ID {
                // If metadata is present, last child is stats table
                if layout.metadata().is_some() && component == (layout.nchildren() - 1) {
                    let metadata_bytes = layout.metadata().expect("extracting stats");
                    let present_stats = stats_from_bitset_bytes(&metadata_bytes[4..]);

                    StatsTable::dtype_for_stats_table(&dtype, &present_stats)
                } else {
                    // If there is no metadata, all children
                    dtype.clone()
                }
            } else if layout.id() == STRUCT_LAYOUT_ID {
                dtype
                    .as_struct()
                    .expect("struct dtype")
                    .field_by_index(component)
                    .expect("struct dtype component access")
            } else if layout.id() == FLAT_LAYOUT_ID {
                // Flat layouts have no children
                unreachable!("flat layouts have no children")
            } else if layout.id() == STATS_LAYOUT_ID {
                if component == 0 {
                    // This is the child data
                    dtype.clone()
                } else {
                    // Otherwise, it's the stats table
                    let present_stats = stats_from_bitset_bytes(
                        &layout.metadata().expect("extracting stats").as_ref()[4..],
                    );
                    StatsTable::dtype_for_stats_table(&dtype, &present_stats)
                }
            } else {
                todo!("unknown DType")
            };

            layout = layout
                .child(component, dtype.clone(), "?")
                .expect("children");
        }

        Self {
            segment_map: Arc::clone(footer.segment_map()),
            path,
            footer,
            layout,
        }
    }

    /// Create a new LayoutCursor indexing into the n-th child of the layout at the current
    /// cursor position.
    pub fn child(&self, n: usize) -> Self {
        let mut path = self.path.clone();
        path.push(n);

        Self::new_with_path(self.footer.clone(), path)
    }

    pub fn parent(&self) -> Self {
        let mut path = self.path.clone();
        path.pop();

        Self::new_with_path(self.footer.clone(), path)
    }

    /// Get the size of the backing flatbuffer for this layout.
    ///
    /// NOTE: this is only safe to run against a FLAT layout.
    pub fn flatbuffer_size(&self) -> usize {
        assert_eq!(
            self.layout.id(),
            FLAT_LAYOUT_ID,
            "flatbuffer size can only be checked for FLAT layout"
        );

        self.layout
            .segments()
            .last()
            .map(|id| self.segment_spec(id).length as usize)
            .unwrap_or_default()
    }

    pub fn total_size(&self) -> usize {
        self.layout_segments()
            .iter()
            .map(|id| self.segment_spec(*id).length as usize)
            .sum()
    }

    fn layout_segments(&self) -> Vec<SegmentId> {
        let segments = collect_segment_ids(&self.layout);
        [segments.0, segments.1].concat()
    }

    /// Predicate true when the cursor is currently activated over a stats table
    pub fn is_stats_table(&self) -> bool {
        let parent = self.parent();
        parent.encoding().id() == STATS_LAYOUT_ID
            && parent.layout().metadata().is_some()
            && self.path.last().copied().unwrap_or_default() == (parent.layout().nchildren() - 1)
    }

    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    pub fn encoding(&self) -> &LayoutVTableRef {
        self.layout.vtable()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn segment_spec(&self, id: SegmentId) -> &SegmentSpec {
        &self.segment_map[*id as usize]
    }
}

#[derive(Default, PartialEq, Eq)]
pub enum KeyMode {
    /// Normal mode.
    ///
    /// The default mode of the TUI when you start it up. Allows for browsing through layout hierarchies.
    #[default]
    Normal,
    /// Searching mode.
    ///
    /// Triggered by a user when entering `/`, subsequent key presses will be used to craft a live-updating filter
    /// of the current input element.
    Search,
}

/// State saved across all Tabs.
///
/// Holding them all allows us to switch between tabs without resetting view state.
pub struct AppState {
    pub key_mode: KeyMode,
    pub search_filter: String,
    pub filter: Option<Vec<bool>>,

    pub footer: Footer,
    pub reader: Arc<dyn AsyncSegmentReader>,
    pub cursor: LayoutCursor,
    pub current_tab: Tab,

    /// List state for the Layouts view
    pub layouts_list_state: ListState,
}

impl AppState {
    pub fn clear_search(&mut self) {
        self.search_filter.clear();
        self.filter.take();
    }
}

/// Create an app backed from a file path.
pub async fn create_file_app(path: impl AsRef<Path>) -> VortexResult<AppState> {
    let file = TokioFile::open(path)?;
    let footer = VortexOpenOptions::file(file.clone())
        .open()
        .await?
        .footer()
        .clone();

    let reader = Arc::new(SegmentReader {
        reader: file.clone(),
        footer: footer.clone(),
    }) as _;

    let cursor = LayoutCursor::new(footer.clone());

    Ok(AppState {
        footer,
        reader,
        cursor,
        key_mode: KeyMode::default(),
        search_filter: String::new(),
        filter: None,
        current_tab: Tab::default(),
        layouts_list_state: ListState::default().with_selected(Some(0)),
    })
}

struct SegmentReader {
    pub reader: TokioFile,
    pub footer: Footer,
}

impl SegmentReader {
    // Read the provided byte range
    fn read_bytes_sync(&self, range: Range<u64>, alignment: Alignment) -> ByteBuffer {
        let mut buf = ByteBufferMut::zeroed_aligned(
            (range.end - range.start).try_into().vortex_expect("range"),
            alignment,
        );
        self.reader
            .read_exact_at(&mut buf, range.start)
            .expect("read_exact_at sync");
        buf.freeze()
    }
}

#[async_trait]
impl AsyncSegmentReader for SegmentReader {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        let segment = &self.footer.segment_map()[*id as usize];
        let range = segment.offset..(segment.offset + segment.length as u64);
        Ok(self.read_bytes_sync(range, segment.alignment))
    }
}

pub fn collect_segment_ids(root_layout: &Layout) -> (Vec<SegmentId>, Vec<SegmentId>) {
    let mut data_segment_ids = Vec::default();
    let mut stats_segment_ids = Vec::default();

    collect_segment_ids_impl(root_layout, &mut data_segment_ids, &mut stats_segment_ids)
        .vortex_unwrap();

    (data_segment_ids, stats_segment_ids)
}

fn collect_segment_ids_impl(
    root: &Layout,
    data_segments: &mut Vec<SegmentId>,
    stats_segments: &mut Vec<SegmentId>,
) -> VortexResult<()> {
    let layout_id = root.id();

    if layout_id == StructLayout.id() {
        let dtype = root.dtype().as_struct().vortex_expect("");
        for child_idx in 0..dtype.fields().len() {
            let name = dtype.field_name(child_idx)?;
            let child_dtype = dtype.field_by_index(child_idx)?;
            let child_layout = root.child(child_idx, child_dtype, name)?;
            collect_segment_ids_impl(&child_layout, data_segments, stats_segments)?;
        }
    } else if layout_id == ChunkedLayout.id() {
        for child_idx in 0..root.nchildren() {
            let child_layout =
                root.child(child_idx, root.dtype().clone(), format!("[{child_idx}]"))?;
            collect_segment_ids_impl(&child_layout, data_segments, stats_segments)?;
        }
    } else if layout_id == StatsLayout.id() {
        let data_layout = root.child(0, root.dtype().clone(), "data")?;
        collect_segment_ids_impl(&data_layout, data_segments, stats_segments)?;

        // For the stats layout, we use the stats segment accumulator
        let stats_layout = root.child(1, root.dtype().clone(), "stats")?;
        collect_segment_ids_impl(&stats_layout, stats_segments, &mut vec![])?;
    } else if layout_id == FlatLayout.id() {
        data_segments.extend(root.segments());
    } else {
        unreachable!()
    };

    Ok(())
}
