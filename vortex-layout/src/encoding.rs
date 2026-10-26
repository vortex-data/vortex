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
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DynLayout;
use crate::LayoutId;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;

/// Backwards-compatible name for a layout ID.
pub type LayoutEncodingId = LayoutId;
/// Shared reference to a registered layout plugin.
pub type LayoutPluginRef = ArcRef<dyn LayoutPlugin>;
/// Backwards-compatible name for a layout plugin reference.
pub type LayoutVTableRef = LayoutPluginRef;
/// Backwards-compatible name for a layout plugin reference.
pub type LayoutEncodingRef = LayoutPluginRef;

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

/// The wire representation of a layout node, chosen by its plugin.
#[derive(Clone, Debug)]
pub struct LayoutSerialization {
    /// The concrete layout ID to write on the wire.
    pub serialized_id: LayoutId,
    /// Layout-specific metadata written into the layout node.
    pub metadata: Vec<u8>,
}

impl LayoutSerialization {
    /// Write `layout` under its own in-memory ID with its vtable's metadata.
    pub fn from_layout(layout: &dyn DynLayout) -> Self {
        Self {
            serialized_id: layout.encoding_id(),
            metadata: layout.metadata(),
        }
    }
}

/// Object-safe plugin registered for a layout ID.
///
/// A plugin has one [`id`](Self::id) for the in-memory layout and one or more
/// [`serialized_ids`](Self::serialized_ids) for wire representations. Its serializer chooses the
/// wire representation, and the layout context validates that the chosen ID is permitted before
/// it is written.
///
/// Every serialized ID is also registered for deserialization, so a plugin may read several
/// historical IDs into the same in-memory layout. A reader that predates a newer ID has no
/// registration for it and reports it as unknown.
pub trait LayoutPlugin: 'static + Send + Sync + Debug {
    /// Returns this plugin as [`Any`].
    fn as_any(&self) -> &dyn Any;
    /// Returns the ID of the in-memory layout handled by this plugin.
    fn id(&self) -> LayoutEncodingId;

    /// Returns the serialized layout IDs understood by this plugin, ordered oldest to newest.
    ///
    /// The default uses the in-memory ID as the sole wire ID.
    fn serialized_ids(&self) -> Vec<LayoutId> {
        vec![self.id()]
    }

    /// Serialize `layout` to its wire representation.
    ///
    /// This function is called only for layouts whose in-memory ID matches [`id`](Self::id). The
    /// returned ID must be declared by [`serialized_ids`](Self::serialized_ids).
    fn serialize(
        &self,
        layout: &dyn DynLayout,
        session: &VortexSession,
    ) -> VortexResult<LayoutSerialization> {
        _ = session;
        Ok(LayoutSerialization::from_layout(layout))
    }

    /// Deserializes a layout node written under `serialized_id`, the exact layout ID found on the
    /// wire.
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        serialized_id: LayoutId,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef>;

    /// Returns `true` if this layout is indivisible: its readers never register natural split
    /// boundaries strictly inside their row range (see [`VTable::is_indivisible`]).
    fn is_indivisible(&self) -> bool {
        false
    }
}

/// Backwards-compatible name for the layout plugin.
pub use LayoutPlugin as LayoutEncoding;
/// Backwards-compatible name for the layout plugin.
pub use LayoutPlugin as LayoutVTablePlugin;

impl<V: VTable> LayoutPlugin for V {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutEncodingId {
        VTable::id(self)
    }

    fn build(
        &self,
        serialized_id: LayoutId,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef> {
        vortex_ensure_eq!(
            VTable::id(self),
            serialized_id,
            "layout plugin does not recognize the serialized ID",
        );
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

    fn is_indivisible(&self) -> bool {
        VTable::is_indivisible(self)
    }
}

impl Display for dyn LayoutPlugin + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn LayoutPlugin + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn LayoutPlugin + '_ {}

impl dyn LayoutPlugin + '_ {
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
