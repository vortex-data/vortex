use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::{LayoutReader, LayoutReaderRef};

pub trait ReaderVisitor {
    /// Visit a child of the reader.
    ///
    /// The `name` provides
    ///
    /// The `field_path` is the path relative to the current [`LayoutReader`] that this child
    /// occupies.
    ///
    /// The `row_offset` indicates the offset of the first row of the child relative to the
    /// current [`LayoutReader`]. This allows us to infer the positions of layout readers within
    /// a dataset, even when [`LayoutReader`]s are concatenated or otherwise combined.
    fn visit_child(
        &self,
        name: &str,
        row_offset: u64,
        field_path: &FieldPath,
        reader: &LayoutReaderRef,
    ) -> VortexResult<()>;
}
