use std::sync::Arc;

use vortex_dtype::{DType, FieldPath};
use vortex_error::vortex_panic;

use crate::LayoutRef;

/// Abstract way of accessing the children of a layout.
///
/// This allows us to abstract over the lazy flatbuffer-based layouts, as well as the in-memory
/// layout trees.
pub trait LayoutChildren: 'static + Send + Sync {
    fn to_arc(&self) -> Arc<dyn LayoutChildren>;

    fn child(&self, idx: usize, dtype: &DType) -> LayoutRef;

    fn child_row_count(&self, idx: usize) -> u64;

    fn nchildren(&self) -> usize;
}

impl LayoutChildren for Arc<dyn LayoutChildren> {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        self.clone()
    }

    fn child(&self, idx: usize, dtype: &DType) -> LayoutRef {
        self.as_ref().child(idx, dtype)
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.as_ref().child_row_count(idx)
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }
}

#[derive(Clone)]
pub struct OwnedLayoutChildren(Vec<LayoutRef>);

impl From<Vec<LayoutRef>> for OwnedLayoutChildren {
    fn from(value: Vec<LayoutRef>) -> Self {
        OwnedLayoutChildren(value)
    }
}

/// In-memory implementation of [`LayoutChildren`].
impl LayoutChildren for OwnedLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> LayoutRef {
        let child = &self.0[idx];
        if child.dtype() != dtype {
            vortex_panic!("Child dtype mismatch: {} != {}", child.dtype(), dtype);
        }
        child.clone()
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.0[idx].row_count()
    }

    fn nchildren(&self) -> usize {
        self.0.len()
    }
}

/// A [`LayoutVisitor`] is a trait that allows us to traverse layouts while maintaining an
/// understanding of their position within the overall dataset.
///
/// Each visit to a child layout provides the relative row offset and field path occupied by
/// that child, as well as a name for debugging and display information.
pub trait LayoutVisitor {
    /// Visit a child of the layout.
    ///
    /// The `name` provides
    ///
    /// The `field_path` is the path relative to the current [`LayoutReader`] that this child
    /// occupies.
    ///
    /// If the child is an auxiliary layout, such as a zone map or dictionary codes, then the
    /// `field_path` should be `None`.
    ///
    /// If the child is a data layout, then the `field_path` should be the path inside the dtype
    /// that this child occupies. If the child does not step into a field, for example the chunks
    /// in a chunked array, then `field_path` should be [`FieldPath::root`].
    ///
    /// The `row_offset` indicates the offset of the first row of the child relative to the
    /// current [`LayoutReader`]. This allows us to infer the positions of layout readers within
    /// a dataset, even when [`LayoutReader`]s are concatenated or otherwise combined.
    fn visit_child(
        &mut self,
        name: &str,
        row_offset: u64,
        // FIXME(ngates): this API is a little wrong, we should fix FieldMask.
        field_path: Option<&FieldPath>,
        child: &LayoutRef,
    );
}
