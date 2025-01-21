use std::collections::BTreeSet;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_array::ContextRef;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;

use crate::segments::AsyncSegmentReader;
use crate::{LayoutData, LayoutReader};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

pub trait LayoutEncoding: Debug + Send + Sync {
    /// Returns the globally unique ID for this type of layout.
    fn id(&self) -> LayoutId;

    /// Construct a [`LayoutReader`] for the provided [`LayoutData`].
    ///
    /// May panic if the provided `LayoutData` is not the same encoding as this `LayoutEncoding`.
    fn reader(
        &self,
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
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
        layout: &LayoutData,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()>;
}

pub type LayoutEncodingRef = &'static dyn LayoutEncoding;
