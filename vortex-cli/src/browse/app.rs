use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use ratatui::widgets::ListState;
use vortex::buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex::dtype::{DType, Field, Nullability, StructDType};
use vortex::error::{VortexExpect, VortexResult};
use vortex::file::{
    ExecutionMode, FileLayout, Segment, SplitBy, VortexOpenOptions, CHUNKED_LAYOUT_ID,
    COLUMNAR_LAYOUT_ID, FLAT_LAYOUT_ID,
};
use vortex::io::TokioFile;
use vortex::sampling_compressor::ALL_ENCODINGS_CONTEXT;
use vortex::stats::stats_from_bitset_bytes;
use vortex_layout::segments::SegmentId;
use vortex_layout::{LayoutData, LayoutEncodingRef};
// Add a shared Tokio Runtime for use in the app.

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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Encoding {
    Flat,
    Chunked,
    Columnar,
    Unknown,
}

impl From<u16> for Encoding {
    fn from(value: u16) -> Self {
        if value == FLAT_LAYOUT_ID.0 {
            Encoding::Flat
        } else if value == CHUNKED_LAYOUT_ID.0 {
            Encoding::Chunked
        } else if value == COLUMNAR_LAYOUT_ID.0 {
            Encoding::Columnar
        } else {
            Encoding::Unknown
        }
    }
}

/// A pointer into the `Layout` hierarchy that can be advanced.
///
/// The pointer wraps an InitialRead.
pub struct LayoutCursor {
    path: Vec<usize>,
    file_layout: FileLayout,
    layout: LayoutData,
    #[allow(unused)]
    segment_map: Arc<[Segment]>,
}

impl LayoutCursor {
    pub fn new(layout: FileLayout) -> Self {
        Self {
            path: Vec::new(),
            layout: layout.root_layout().clone(),
            segment_map: Arc::clone(layout.segment_map()),
            file_layout: layout,
        }
    }

    pub fn new_with_path(file_layout: FileLayout, path: Vec<usize>) -> Self {
        let mut layout = file_layout.root_layout().clone();
        let mut dtype = file_layout.dtype().clone();
        // Traverse the layout tree at each element of the path.
        for component in path.iter().copied() {
            // Find the DType of the child based on the DType of the current node.
            dtype = match layout.encoding().id() {
                CHUNKED_LAYOUT_ID => {
                    // If metadata is present, last child is stats table
                    if layout.metadata().is_some() && component == (layout.nchildren() - 1) {
                        let stats = stats_from_bitset_bytes(
                            layout.metadata().expect("extracting stats").as_ref(),
                        );

                        // When Chunked layout has a metadata field set, it will have a DType with
                        // STRUCT type and one field for each of the statistics.
                        let struct_dtype = StructDType::new(
                            stats
                                .iter()
                                .map(|stat| Arc::from(stat.to_string().as_str()))
                                .collect::<Vec<Arc<str>>>()
                                .into(),
                            stats
                                .iter()
                                .map(|stat| stat.dtype(&dtype))
                                .collect::<Vec<DType>>(),
                        );
                        DType::Struct(Arc::new(struct_dtype), Nullability::NonNullable)
                    } else {
                        // If there is no metadata, all children
                        dtype.clone()
                    }
                }
                COLUMNAR_LAYOUT_ID => dtype
                    .as_struct()
                    .expect("struct dtype")
                    .field_info(&Field::Index(component))
                    .expect("struct dtype component access")
                    .dtype
                    .value()
                    .expect("dtype value"),
                // Flat layouts have no children
                FLAT_LAYOUT_ID => unreachable!("flat layouts have no children"),
                _ => todo!("unknown DType"),
            };

            layout = layout.child(component, dtype.clone()).expect("children");
        }

        Self {
            segment_map: Arc::clone(file_layout.segment_map()),
            path,
            file_layout,
            layout,
        }
    }

    /// Create a new LayoutCursor indexing into the n-th child of the layout at the current
    /// cursor position.
    pub fn child(&self, n: usize) -> Self {
        let mut path = self.path.clone();
        path.push(n);

        Self::new_with_path(self.file_layout.clone(), path)
    }

    pub fn parent(&self) -> Self {
        let mut path = self.path.clone();
        path.pop();

        Self::new_with_path(self.file_layout.clone(), path)
    }

    /// Predicate true when the cursor is currently activated over a stats table
    pub fn is_stats_table(&self) -> bool {
        let parent = self.parent();
        parent.encoding().id() == CHUNKED_LAYOUT_ID
            && self.path.last().copied().unwrap_or_default() == (parent.layout().nchildren() - 1)
    }

    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    pub fn encoding(&self) -> LayoutEncodingRef {
        self.layout.encoding()
    }

    pub fn layout(&self) -> &LayoutData {
        &self.layout
    }

    pub fn segment(&self, id: SegmentId) -> &Segment {
        &self.segment_map[*id as usize]
    }
}

/// State saved across all Tabs.
///
/// Holding them all allows us to switch between tabs without resetting view state.
pub struct AppState {
    pub reader: TokioFile,
    pub cursor: LayoutCursor,
    pub current_tab: Tab,

    /// List state for the Layouts view
    pub layouts_list_state: ListState,
}

impl AppState {
    pub fn read_segment(&self, segment_id: SegmentId) -> ByteBuffer {
        let segment = self.cursor.segment(segment_id);
        let range = segment.offset..(segment.offset + segment.length as u64);
        self.read_bytes_sync(range, segment.alignment)
    }

    // Read the provided byte range
    pub fn read_bytes_sync(&self, range: Range<u64>, alignment: Alignment) -> ByteBuffer {
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

/// Create an app backed from a file path.
pub async fn create_file_app(path: impl AsRef<Path>) -> VortexResult<AppState> {
    let reader = TokioFile::open(path)?;
    let file = VortexOpenOptions::new(ALL_ENCODINGS_CONTEXT.clone())
        .with_execution_mode(ExecutionMode::Inline)
        .with_split_by(SplitBy::Layout)
        .open(reader.clone())
        .await?;

    let cursor = LayoutCursor::new(file.file_layout().clone());

    Ok(AppState {
        reader,
        cursor,
        current_tab: Tab::default(),
        layouts_list_state: ListState::default().with_selected(Some(0)),
    })
}
