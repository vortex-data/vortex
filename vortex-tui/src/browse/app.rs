use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use ratatui::widgets::ListState;
use vortex::buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult};
use vortex::file::{Footer, Segment, VortexOpenOptions};
use vortex::io::TokioFile;
use vortex::stats::stats_from_bitset_bytes;
use vortex_layout::layouts::stats::stats_table::StatsTable;
use vortex_layout::segments::{PendingSegment, SegmentId, SegmentReader};
use vortex_layout::{
    CHUNKED_LAYOUT_ID, FLAT_LAYOUT_ID, Layout, LayoutVTableRef, STATS_LAYOUT_ID, STRUCT_LAYOUT_ID,
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
    #[allow(unused)]
    segment_map: Arc<[Segment]>,
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
                    let present_stats = stats_from_bitset_bytes(
                        layout.metadata().expect("extracting stats").as_ref(),
                    );

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
            .map(|id| self.segment(id).length as usize)
            .unwrap_or_default()
    }

    pub fn segment_size(&self) -> usize {
        self.layout()
            .segments()
            .map(|id| self.segment(id).length as usize)
            .sum()
    }

    /// Predicate true when the cursor is currently activated over a stats table
    pub fn is_stats_table(&self) -> bool {
        let parent = self.parent();
        parent.encoding().id() == CHUNKED_LAYOUT_ID
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

    pub fn segment(&self, id: SegmentId) -> &Segment {
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
    pub reader: Arc<dyn SegmentReader>,
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

    let reader = Arc::new(TuiSegmentReader {
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

struct TuiSegmentReader {
    pub reader: TokioFile,
    pub footer: Footer,
}

impl TuiSegmentReader {
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

impl SegmentReader for TuiSegmentReader {
    fn get(&self, id: SegmentId) -> VortexResult<Arc<dyn PendingSegment>> {
        let segment = &self.footer.segment_map()[*id as usize];
        let range = segment.offset..(segment.offset + segment.length as u64);
        Ok(Arc::new(self.read_bytes_sync(range, segment.alignment)))
    }
}
