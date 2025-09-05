// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::sync::Arc;

use futures::executor::block_on;
use ratatui::prelude::Size;
use ratatui::widgets::ListState;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex::file::{Footer, SegmentSpec, VortexFile, VortexOpenOptions};
use vortex::layout::LayoutRef;
use vortex::layout::layouts::flat::FlatVTable;
use vortex::layout::layouts::zoned::ZonedVTable;
use vortex::layout::segments::{SegmentId, SegmentSource};
use vortex::serde::ArrayParts;

use crate::browse::ui::SegmentGridState;

#[derive(Default, Copy, Clone, Eq, PartialEq)]
pub enum Tab {
    /// The layout tree browser.
    #[default]
    Layout,

    /// Show a segment map of the file
    Segments,
    // TODO(aduffy): SQL query page powered by DF
    // Query,
}

/// A pointer into the `Layout` hierarchy that can be advanced.
///
/// The pointer wraps an InitialRead.
pub struct LayoutCursor {
    path: Vec<usize>,
    footer: Footer,
    layout: LayoutRef,
    segment_map: Arc<[SegmentSpec]>,
    segment_source: Arc<dyn SegmentSource>,
}

impl LayoutCursor {
    pub fn new(footer: Footer, segment_source: Arc<dyn SegmentSource>) -> Self {
        Self {
            path: Vec::new(),
            layout: footer.layout().clone(),
            segment_map: Arc::clone(footer.segment_map()),
            footer,
            segment_source,
        }
    }

    pub fn new_with_path(
        footer: Footer,
        segment_source: Arc<dyn SegmentSource>,
        path: Vec<usize>,
    ) -> Self {
        let mut layout = footer.layout().clone();

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

    /// Create a new LayoutCursor indexing into the n-th child of the layout at the current
    /// cursor position.
    pub fn child(&self, n: usize) -> Self {
        let mut path = self.path.clone();
        path.push(n);

        Self::new_with_path(self.footer.clone(), self.segment_source.clone(), path)
    }

    pub fn parent(&self) -> Self {
        let mut path = self.path.clone();
        path.pop();

        Self::new_with_path(self.footer.clone(), self.segment_source.clone(), path)
    }

    /// Get the size of the array flatbuffer for this layout.
    ///
    /// NOTE: this is only safe to run against a FLAT layout.
    pub fn flatbuffer_size(&self) -> usize {
        let segment_id = self.layout.as_::<FlatVTable>().segment_id();
        let segment = block_on(self.segment_source.request(segment_id)).vortex_unwrap();
        ArrayParts::try_from(segment)
            .vortex_unwrap()
            .metadata()
            .len()
    }

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

    /// Predicate true when the cursor is currently activated over a stats table
    pub fn is_stats_table(&self) -> bool {
        let parent = self.parent();
        parent.layout().is::<ZonedVTable>() && self.path.last().copied().unwrap_or_default() == 1
    }

    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    pub fn layout(&self) -> &LayoutRef {
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
pub struct AppState<'a> {
    pub key_mode: KeyMode,
    pub search_filter: String,
    pub filter: Option<Vec<bool>>,

    pub vxf: VortexFile,
    pub cursor: LayoutCursor,
    pub current_tab: Tab,

    /// List state for the Layouts view
    pub layouts_list_state: ListState,
    pub segment_grid_state: SegmentGridState<'a>,
    pub frame_size: Size,
}

impl AppState<'_> {
    pub fn clear_search(&mut self) {
        self.search_filter.clear();
        self.filter.take();
    }
}

/// Create an app backed from a file path.
pub async fn create_file_app<'a>(path: impl AsRef<Path>) -> VortexResult<AppState<'a>> {
    let vxf = VortexOpenOptions::file().open(path).await?;

    let cursor = LayoutCursor::new(vxf.footer().clone(), vxf.segment_source());

    Ok(AppState {
        vxf,
        cursor,
        key_mode: KeyMode::default(),
        search_filter: String::new(),
        filter: None,
        current_tab: Tab::default(),
        layouts_list_state: ListState::default().with_selected(Some(0)),
        segment_grid_state: SegmentGridState::default(),
        frame_size: Size::new(0, 0),
    })
}
