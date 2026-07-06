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
use vortex_error::vortex_panic;
use vortex_session::registry::Id;

use crate::IntoLayout;
use crate::LayoutBuildContext;
use crate::LayoutChildren;
use crate::LayoutRef;
use crate::VTable;
use crate::segments::SegmentId;

/// A unique identifier for a layout encoding.
pub type LayoutEncodingId = Id;
/// Shared reference to a registered layout encoding.
pub type LayoutEncodingRef = ArcRef<dyn LayoutEncoding>;

/// Object-safe layout encoding registered in a [`LayoutSession`](crate::session::LayoutSession).
///
/// Encoding instances deserialize serialized layout metadata into concrete [`LayoutRef`] nodes.
/// New in-tree encodings usually implement [`VTable`] and use [`LayoutEncodingAdapter`], while
/// foreign encodings can provide an object-safe implementation directly.
pub trait LayoutEncoding: 'static + Send + Sync + Debug + private::Sealed {
    /// Returns this encoding as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the globally unique encoding id.
    fn id(&self) -> LayoutEncodingId;

    /// Build a layout from serialized metadata, segment ids, and children.
    ///
    /// Implementations must use `build_ctx` for session-scoped plugin resolution instead of global
    /// registries.
    fn build(
        &self,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef>;
}

/// Object-safe adapter from a typed layout [`VTable`] to [`LayoutEncoding`].
#[repr(transparent)]
pub struct LayoutEncodingAdapter<V: VTable>(V::Encoding);

impl<V: VTable> LayoutEncoding for LayoutEncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutEncodingId {
        V::id(&self.0)
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
        let layout = V::build(
            &self.0,
            dtype,
            row_count,
            &metadata,
            segment_ids,
            children,
            build_ctx,
        )?;

        // Validate that the builder function returned the expected values.
        if layout.row_count() != row_count {
            vortex_panic!(
                "Layout row count mismatch: {} != {}",
                layout.row_count(),
                row_count
            );
        }
        if layout.dtype() != dtype {
            vortex_panic!("Layout dtype mismatch: {} != {}", layout.dtype(), dtype);
        }

        Ok(layout.into_layout())
    }
}

impl<V: VTable> Debug for LayoutEncodingAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutEncoding")
            .field("id", &self.id())
            .finish()
    }
}

impl Display for dyn LayoutEncoding + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn LayoutEncoding + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn LayoutEncoding + '_ {}

impl dyn LayoutEncoding + '_ {
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    pub fn as_<V: VTable>(&self) -> &V::Encoding {
        self.as_opt::<V>()
            .vortex_expect("LayoutEncoding is not of the expected type")
    }

    pub fn as_opt<V: VTable>(&self) -> Option<&V::Encoding> {
        self.as_any()
            .downcast_ref::<LayoutEncodingAdapter<V>>()
            .map(|e| &e.0)
    }
}

mod private {
    use super::*;
    use crate::layouts::foreign::ForeignLayoutEncoding;

    pub trait Sealed {}

    impl<V: VTable> Sealed for LayoutEncodingAdapter<V> {}
    impl Sealed for ForeignLayoutEncoding {}
}
