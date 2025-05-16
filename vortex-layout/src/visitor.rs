use vortex_dtype::FieldPath;
use vortex_error::VortexResult;

use crate::LayoutRef;

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
    ) -> VortexResult<()>;
}
