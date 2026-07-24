// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    DynLayout, Layout, LayoutBuildContext, LayoutChildType, LayoutChildren, LayoutDeserializeArgs,
    LayoutId, LayoutParts, LayoutReaderContext, LayoutReaderRef,
};
use std::fmt::Debug;
use std::sync::Arc;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

/// Shared, erased handle to a layout tree node.
pub type LayoutRef = Arc<dyn DynLayout>;

/// Layout-specific behavior for a typed [`Layout`].
///
/// Common serialized fields are stored by [`Layout`]. Implementations own only their
/// layout-specific data, metadata codec, child typing, and reader construction.
pub trait VTable: 'static + Clone + Send + Sync + Debug {
    /// Layout-specific data.
    type LayoutData: 'static + Clone + Send + Sync + Debug;
    /// Serialized metadata type.
    type Metadata: SerializeMetadata + vortex_array::DeserializeMetadata + Debug;

    /// Returns the globally unique layout ID.
    fn id(&self) -> LayoutId;

    /// Returns the serializable metadata for a layout.
    fn metadata(layout: &Layout<Self>) -> Self::Metadata;

    /// Deserialize and validate layout-specific data.
    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &<Self::Metadata as vortex_array::DeserializeMetadata>::Output,
    ) -> VortexResult<Self::LayoutData>;

    /// Construct a typed layout from deserialized common fields.
    fn build(
        vtable: &Self,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as vortex_array::DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Layout<Self>> {
        let args = LayoutDeserializeArgs {
            session: build_ctx.session,
            array_read_ctx: build_ctx.array_read_ctx,
            dtype,
            row_count,
            segment_ids,
            children,
        };
        let data = vtable.deserialize(&args, metadata)?;
        Ok(LayoutParts::new(
            vtable.clone(),
            dtype.clone(),
            row_count,
            args.segment_ids,
            children.to_arc(),
            data,
        )
        .into_typed())
    }

    /// Returns the expected dtype of child `idx`.
    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType>;

    /// Returns the relationship between child `idx` and its parent.
    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType;

    /// Construct a reader for this layout.
    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef>;
}
