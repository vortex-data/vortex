// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use arcref::ArcRef;
use vortex_array::DeserializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutId;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;

/// Backwards-compatible name for a layout ID.
pub type LayoutEncodingId = LayoutId;
/// Shared reference to a registered layout-vtable plugin.
pub type LayoutVTableRef = ArcRef<dyn LayoutVTablePlugin>;
/// Backwards-compatible name for a layout-vtable reference.
pub type LayoutEncodingRef = LayoutVTableRef;

/// Common fields available while deserializing layout-specific data.
pub struct LayoutDeserializeArgs<'a> {
    /// Session used to resolve plugin-owned metadata.
    pub session: &'a VortexSession,
    /// Array read context referenced by serialized array metadata.
    pub array_read_ctx: &'a ReadContext,
    /// Logical dtype of this layout.
    pub dtype: &'a DType,
    /// Number of rows in this layout.
    pub row_count: u64,
    /// Directly referenced segments.
    pub segment_ids: Vec<SegmentId>,
    /// Lazy child access.
    pub children: &'a dyn LayoutChildren,
}

/// Context shared while recursively deserializing layouts.
pub struct LayoutBuildContext<'a> {
    /// Session used to resolve plugin-owned metadata.
    pub session: &'a VortexSession,
    /// Array read context referenced by serialized array metadata.
    pub array_read_ctx: &'a ReadContext,
}

/// Object-safe plugin registered for a layout ID.
pub trait LayoutVTablePlugin: 'static + Send + Sync + Debug {
    /// Returns this plugin as [`Any`].
    fn as_any(&self) -> &dyn Any;
    /// Returns the globally unique layout ID.
    fn id(&self) -> LayoutEncodingId;
    /// Deserializes a layout node.
    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef>;

    /// Returns `true` if readers of this layout may register natural split boundaries strictly
    /// inside their row range (see [`VTable::registers_interior_splits`]).
    fn registers_interior_splits(&self) -> bool {
        true
    }
}

/// Backwards-compatible name for the object-safe layout-vtable plugin.
pub use LayoutVTablePlugin as LayoutEncoding;

impl<V: VTable> LayoutVTablePlugin for V {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutEncodingId {
        VTable::id(self)
    }

    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef> {
        let metadata = <V::Metadata as DeserializeMetadata>::deserialize(metadata)?;
        Ok(V::build(
            self,
            dtype,
            row_count,
            &metadata,
            segment_ids,
            children,
            build_ctx,
        )?
        .into_layout())
    }

    fn registers_interior_splits(&self) -> bool {
        VTable::registers_interior_splits(self)
    }
}

impl Display for dyn LayoutVTablePlugin + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn LayoutVTablePlugin + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn LayoutVTablePlugin + '_ {}

impl dyn LayoutVTablePlugin + '_ {
    /// Returns whether this plugin is vtable `V`.
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Downcasts this plugin to vtable `V`.
    pub fn as_<V: VTable>(&self) -> &V {
        self.as_opt::<V>()
            .vortex_expect("layout encoding type mismatch")
    }

    /// Attempts to downcast this plugin to vtable `V`.
    pub fn as_opt<V: VTable>(&self) -> Option<&V> {
        self.as_any().downcast_ref()
    }
}
