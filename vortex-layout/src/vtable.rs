use std::collections::BTreeSet;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::scan::ScanExecutor;
use crate::{Layout, LayoutId, LayoutReader};

/// A reference to a layout VTable, either static or arc'd.
#[derive(Debug, Clone)]
pub struct LayoutVTableRef(Inner);

#[derive(Debug, Clone)]
enum Inner {
    Static(&'static dyn LayoutVTable),
    Arc(Arc<dyn LayoutVTable>),
}

impl LayoutVTableRef {
    pub const fn from_static(vtable: &'static dyn LayoutVTable) -> Self {
        LayoutVTableRef(Inner::Static(vtable))
    }

    pub fn from_arc(vtable: Arc<dyn LayoutVTable>) -> Self {
        LayoutVTableRef(Inner::Arc(vtable))
    }
}

impl Deref for LayoutVTableRef {
    type Target = dyn LayoutVTable;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            Inner::Static(vtable) => *vtable,
            Inner::Arc(vtable) => vtable.deref(),
        }
    }
}

pub trait LayoutVTable: Debug + Send + Sync {
    /// Returns the globally unique ID for this type of layout.
    fn id(&self) -> LayoutId;

    /// Construct a [`LayoutReader`] for the provided [`Layout`].
    ///
    /// May panic if the provided `Layout` is not the same encoding as this `LayoutEncoding`.
    fn reader(
        &self,
        layout: Layout,
        ctx: ContextRef,
        executor: Arc<ScanExecutor>,
    ) -> VortexResult<Arc<dyn LayoutReader>>;

    /// Register the row splits for this layout, these represent natural boundaries at which
    /// a reader can split the layout for independent processing.
    ///
    /// For example, a ChunkedLayout would register a boundary at the end of every chunk.
    ///
    /// The layout is passed a `row_offset` that identifies the starting row of the layout within
    /// the file.
    // TODO(ngates): we should check whether this is actually performant enough since we visit
    //  all nodes of the layout tree, often registering the same splits many times.
    fn register_splits(
        &self,
        layout: &Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;
}
