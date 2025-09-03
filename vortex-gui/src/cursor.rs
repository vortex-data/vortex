// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use tokio::runtime::Handle;
use vortex::dtype::DType;
use vortex::file::{Footer, SegmentSpec};
use vortex::layout::LayoutRef;
use vortex::layout::layouts::flat::FlatVTable;
use vortex::layout::layouts::zoned::ZonedVTable;
use vortex::layout::segments::{SegmentId, SegmentSource};
use vortex::serde::ArrayParts;

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
            layout = layout.child(component).unwrap();
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
        let segment = Handle::current()
            .block_on(self.segment_source.request(segment_id))
            .unwrap();
        ArrayParts::try_from(segment).unwrap().metadata().len()
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
            .map(|layout| layout.expect("Failed to load layout"))
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
