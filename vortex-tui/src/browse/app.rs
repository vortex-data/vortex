// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Application state and data structures for the TUI browser.

use std::sync::Arc;

use ratatui::prelude::Size;
use ratatui::widgets::ListState;
use vortex::array::ArrayRef;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::file::Footer;
use vortex::file::SegmentSpec;
use vortex::file::VortexFile;
use vortex::layout::LayoutRef;
use vortex::layout::VTable;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::session::VortexSession;

use super::ui::SegmentGridState;

/// The currently active tab in the TUI browser.
#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub enum Tab {
    /// The layout tree browser tab.
    ///
    /// Shows the hierarchical structure of layouts in the Vortex file and allows navigation
    /// through the layout tree.
    #[default]
    Layout,

    /// The segment map tab.
    ///
    /// Displays a visual representation of how segments are laid out in the file.
    Segments,

    /// SQL query interface powered by DataFusion.
    #[cfg(feature = "native")]
    Query,
}

/// A navigable pointer into the layout hierarchy of a Vortex file.
///
/// The cursor maintains the current position within the layout tree and provides methods to
/// navigate up and down the hierarchy. It also provides access to layout metadata and segment
/// information at the current position.
pub struct LayoutCursor {
    path: Vec<usize>,
    footer: Footer,
    layout: LayoutRef,
    segment_map: Arc<[SegmentSpec]>,
    segment_source: Arc<dyn SegmentSource>,
}

impl LayoutCursor {
    /// Create a new cursor pointing at the root layout.
    pub fn new(footer: Footer, segment_source: Arc<dyn SegmentSource>) -> Self {
        Self {
            path: Vec::new(),
            layout: Arc::clone(footer.layout()),
            segment_map: Arc::clone(footer.segment_map()),
            footer,
            segment_source,
        }
    }

    /// Create a new cursor at a specific path within the layout tree.
    ///
    /// The path is a sequence of child indices to traverse from the root.
    pub fn new_with_path(
        footer: Footer,
        segment_source: Arc<dyn SegmentSource>,
        path: Vec<usize>,
    ) -> Self {
        let mut layout = Arc::clone(footer.layout());

        // Traverse the layout tree at each element of the path.
        for component in path.iter().copied() {
            layout = layout
                .child(component)
                .vortex_expect("Failed to get child layout");
        }

        Self {
            segment_map: Arc::clone(footer.segment_map()),
            path,
            footer,
            layout,
            segment_source,
        }
    }

    /// Create a new cursor pointing at the n-th child of the current layout.
    pub fn child(&self, n: usize) -> Self {
        let mut path = self.path.clone();
        path.push(n);

        Self::new_with_path(self.footer.clone(), Arc::clone(&self.segment_source), path)
    }

    /// Create a new cursor pointing at the parent of the current layout.
    ///
    /// If already at the root, returns a cursor pointing at the root.
    pub fn parent(&self) -> Self {
        let mut path = self.path.clone();
        path.pop();

        Self::new_with_path(self.footer.clone(), Arc::clone(&self.segment_source), path)
    }

    /// Get a human-readable description of the flat layout metadata.
    ///
    /// # Panics
    ///
    /// Panics if the current layout is not a [`Flat`] layout.
    pub fn flat_layout_metadata_info(&self) -> String {
        let flat_layout = self.layout.as_::<Flat>();
        let metadata = Flat::metadata(flat_layout);

        match metadata.0.array_encoding_tree.as_ref() {
            Some(tree) => {
                let size = tree.len();
                format!(
                    "Flat Metadata: array_encoding_tree present ({} bytes)",
                    size
                )
            }
            None => "Flat Metadata: array_encoding_tree not present".to_string(),
        }
    }

    /// Get the total size in bytes of all segments reachable from this layout.
    pub fn total_size(&self) -> usize {
        self.layout_segments()
            .iter()
            .map(|id| self.segment_spec(*id).length as usize)
            .sum()
    }

    fn layout_segments(&self) -> Vec<SegmentId> {
        self.layout
            .depth_first_traversal()
            .map(|layout| layout.vortex_expect("Failed to load layout"))
            .flat_map(|layout| layout.segment_ids().into_iter())
            .collect()
    }

    /// Returns `true` if the cursor is currently pointing at a statistics table.
    ///
    /// A statistics table is the second child of a [`Zoned`] layout.
    pub fn is_stats_table(&self) -> bool {
        let parent = self.parent();
        parent.layout().is::<Zoned>() && self.path.last().copied().unwrap_or_default() == 1
    }

    /// Get the data type of the current layout.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Get a reference to the current layout.
    pub fn layout(&self) -> &LayoutRef {
        &self.layout
    }

    /// Get the segment specification for a given segment ID.
    pub fn segment_spec(&self, id: SegmentId) -> &SegmentSpec {
        &self.segment_map[*id as usize]
    }
}

/// The current input mode of the TUI.
///
/// Different modes change how keyboard input is interpreted.
#[derive(Default, PartialEq, Eq)]
pub enum KeyMode {
    /// Normal navigation mode.
    ///
    /// The default mode when the TUI starts. Allows browsing through the layout hierarchy using
    /// arrow keys, vim-style navigation (`h`/`j`/`k`/`l`), and various shortcuts.
    #[default]
    Normal,

    /// Search/filter mode.
    ///
    /// Activated by pressing `/` or `Ctrl-S`. In this mode, key presses are used to build a fuzzy
    /// search filter that narrows down the displayed layout children. Press `Esc` or `Ctrl-G` to
    /// exit search mode.
    Search,
}

/// The complete application state for the TUI browser.
///
/// This struct holds all state needed to render and interact with the TUI, including:
/// - The Vortex session and file being browsed
/// - Navigation state (current cursor position, selected tab)
/// - Input mode and search filter state
/// - UI state for lists and grids
///
/// The state is preserved when switching between tabs, allowing users to return to their previous
/// position.
pub struct AppState {
    /// The Vortex session used to read array data during rendering.
    pub session: VortexSession,

    /// The current input mode (normal navigation or search).
    pub key_mode: KeyMode,

    /// The current search filter string (only used in search mode).
    pub search_filter: String,

    /// A boolean mask indicating which children match the current search filter.
    ///
    /// `None` when no filter is active, `Some(vec)` when filtering where `vec[i]` indicates
    /// whether child `i` should be shown.
    pub filter: Option<Vec<bool>>,

    /// The open Vortex file being browsed.
    pub vxf: VortexFile,

    /// The current position in the layout hierarchy.
    pub cursor: LayoutCursor,

    /// The currently selected tab.
    pub current_tab: Tab,

    /// Selection state for the layout children list.
    pub layouts_list_state: ListState,

    /// State for the segment grid display.
    pub segment_grid_state: SegmentGridState,

    /// The size of the last rendered frame.
    pub frame_size: Size,

    /// Vertical scroll offset for the encoding tree display in flat layout view.
    pub tree_scroll_offset: u16,

    /// Cached array data for the current FlatLayout, loaded asynchronously on WASM.
    pub cached_flat_array: Option<ArrayRef>,

    /// Cached flatbuffer size for the current FlatLayout, loaded asynchronously on WASM.
    pub cached_flatbuffer_size: Option<usize>,

    /// State for the Query tab.
    #[cfg(feature = "native")]
    pub query_state: super::ui::QueryState,

    /// File path for use in query execution.
    #[cfg(feature = "native")]
    pub file_path: String,
}

impl AppState {
    /// Create a new application state by opening a Vortex file from a path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    #[cfg(feature = "native")]
    pub async fn new(
        session: &VortexSession,
        path: impl AsRef<std::path::Path>,
    ) -> vortex::error::VortexResult<AppState> {
        use vortex::file::OpenOptionsSessionExt;

        let session = session.clone();
        let vxf = session.open_options().open_path(path.as_ref()).await?;

        let cursor = LayoutCursor::new(vxf.footer().clone(), vxf.segment_source());

        let file_path = path
            .as_ref()
            .to_str()
            .map(|s| s.to_string())
            .unwrap_or_default();

        Ok(AppState {
            session,
            vxf,
            cursor,
            key_mode: KeyMode::default(),
            search_filter: String::new(),
            filter: None,
            current_tab: Tab::default(),
            layouts_list_state: ListState::default().with_selected(Some(0)),
            segment_grid_state: SegmentGridState::default(),
            frame_size: Size::new(0, 0),
            tree_scroll_offset: 0,
            cached_flat_array: None,
            cached_flatbuffer_size: None,
            query_state: super::ui::QueryState::default(),
            file_path,
        })
    }

    /// Create a new application state from an in-memory buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer does not contain a valid Vortex file.
    #[cfg(target_arch = "wasm32")]
    pub fn from_buffer(
        session: VortexSession,
        buffer: vortex::buffer::ByteBuffer,
    ) -> vortex::error::VortexResult<AppState> {
        use vortex::file::OpenOptionsSessionExt;

        let vxf = session.open_options().open_buffer(buffer)?;
        let cursor = LayoutCursor::new(vxf.footer().clone(), vxf.segment_source());

        Ok(AppState {
            session,
            vxf,
            cursor,
            key_mode: KeyMode::default(),
            search_filter: String::new(),
            filter: None,
            current_tab: Tab::default(),
            layouts_list_state: ListState::default().with_selected(Some(0)),
            segment_grid_state: SegmentGridState::default(),
            frame_size: Size::new(0, 0),
            tree_scroll_offset: 0,
            cached_flat_array: None,
            cached_flatbuffer_size: None,
        })
    }

    /// Clear the current search filter and return to showing all children.
    pub fn clear_search(&mut self) {
        self.search_filter.clear();
        self.filter.take();
    }

    /// Reset the layout view state after navigating to a different layout.
    ///
    /// This resets the list selection to the first item and clears any scroll offset.
    /// The caller is responsible for awaiting `load_flat_data()` afterward if the
    /// new layout is a [`Flat`].
    pub fn reset_layout_view_state(&mut self) {
        self.layouts_list_state = ListState::default().with_selected(Some(0));
        self.tree_scroll_offset = 0;
        self.cached_flat_array = None;
        self.cached_flatbuffer_size = None;
    }

    /// Asynchronously load and cache the flat layout array data and flatbuffer size.
    #[cfg(feature = "native")]
    pub(crate) async fn load_flat_data(&mut self) {
        use vortex::array::MaskFuture;
        use vortex::array::serde::SerializedArray;
        use vortex::expr::root;

        let layout = &Arc::clone(self.cursor.layout());
        let row_count = layout.row_count();

        // Load the array.
        let reader = layout
            .new_reader("".into(), self.vxf.segment_source(), &self.session)
            .vortex_expect("Failed to create reader");
        let array = reader
            .projection_evaluation(
                &(0..row_count),
                &root(),
                MaskFuture::new_true(
                    usize::try_from(row_count).vortex_expect("row_count overflowed usize"),
                ),
            )
            .vortex_expect("Failed to construct projection")
            .await
            .vortex_expect("Failed to read flat array");
        self.cached_flat_array = Some(array);

        // Load the flatbuffer size.
        let segment_id = layout.as_::<Flat>().segment_id();
        let segment = self
            .cursor
            .segment_source
            .request(segment_id)
            .await
            .vortex_expect("Failed to read segment");
        self.cached_flatbuffer_size = Some(
            SerializedArray::try_from(segment)
                .vortex_expect("Failed to parse segment")
                .metadata()
                .len(),
        );
    }
}
