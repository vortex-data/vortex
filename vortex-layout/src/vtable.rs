// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::IntoLayout;
use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutEncoding;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Context available while constructing a layout from serialized metadata.
pub struct LayoutBuildContext<'a> {
    /// The session used to resolve plugin-owned metadata such as aggregate function options.
    pub session: &'a VortexSession,
    /// The array read context referenced by serialized array metadata in descendant layouts.
    pub array_read_ctx: &'a ReadContext,
}

/// Typed implementation contract for a layout encoding.
///
/// A layout vtable connects a concrete layout node type, its registered encoding object, and the
/// metadata representation used for serialization. The object-safe [`Layout`] and
/// [`LayoutEncoding`] APIs delegate to this trait through adapters.
pub trait VTable: 'static + Sized + Send + Sync + Debug {
    /// Concrete layout node type for this encoding.
    type Layout: 'static + Send + Sync + Clone + Debug + Deref<Target = dyn Layout> + IntoLayout;
    /// Concrete encoding object registered in the session.
    type Encoding: 'static + Send + Sync + Deref<Target = dyn LayoutEncoding>;
    /// Serialized layout metadata type.
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    /// Returns the ID of the layout encoding.
    fn id(encoding: &Self::Encoding) -> LayoutId;

    /// Returns the encoding for the layout.
    fn encoding(layout: &Self::Layout) -> LayoutEncodingRef;

    /// Returns the row count for the layout.
    fn row_count(layout: &Self::Layout) -> u64;

    /// Returns the dtype for the layout reader.
    fn dtype(layout: &Self::Layout) -> &DType;

    /// Returns the metadata for the layout.
    fn metadata(layout: &Self::Layout) -> Self::Metadata;

    /// Returns the segment IDs for the layout.
    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId>;

    /// Returns the number of children for the layout.
    fn nchildren(layout: &Self::Layout) -> usize;

    /// Return the child at the given index.
    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef>;

    /// Return the type of the child at the given index.
    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType;

    /// Create a new reader for the layout.
    ///
    /// **Layouts with children MUST propagate `ctx` to descendants** by passing it
    /// through `Layout::new_reader` (or `LazyReaderChildren::new`) when constructing
    /// child readers. If `ctx` is dropped at any link in the chain, ancestor-published
    /// values won't reach affected descendants — a silent runtime regression for any
    /// descendant that looked up an ancestor-published value via `ctx.get::<T>()`.
    /// There is no compile-time check that catches this; reviewer discipline + the
    /// integration tests in `vortex-layout` are the only safety net.
    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef>;

    /// Construct a new [`Layout`] from deserialized parts.
    ///
    /// Implementations should validate child count, child types, row counts, segment references,
    /// and dtype consistency for their encoding. The generic adapter checks the returned layout's
    /// top-level dtype and row count, but encoding-specific invariants belong here.
    fn build(
        encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout>;

    /// Replaces the children of the layout with the given layout references.
    ///
    /// The count and types of children must match the layout's requirements.
    /// This method is used for transforming layout trees by replacing child layouts.
    fn with_children(_layout: &mut Self::Layout, _children: Vec<LayoutRef>) -> VortexResult<()> {
        vortex_bail!("with_children not implemented for this layout")
    }
}

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct $V;

            impl AsRef<dyn $crate::Layout> for [<$V Layout>] {
                fn as_ref(&self) -> &dyn $crate::Layout {
                    // SAFETY: LayoutAdapter is #[repr(transparent)] over the Layout type,
                    // which guarantees identical memory layout. This cast is safe because
                    // we're only changing the type metadata, not the actual data.
                    unsafe { &*(self as *const [<$V Layout>] as *const $crate::LayoutAdapter<$V>) }
                }
            }

            impl std::ops::Deref for [<$V Layout>] {
                type Target = dyn $crate::Layout;

                fn deref(&self) -> &Self::Target {
                    // SAFETY: LayoutAdapter is #[repr(transparent)] over the Layout type,
                    // which guarantees identical memory layout. This cast is safe because
                    // we're only changing the type metadata, not the actual data.
                    unsafe { &*(self as *const [<$V Layout>] as *const $crate::LayoutAdapter<$V>) }
                }
            }

            impl $crate::IntoLayout for [<$V Layout>] {
                fn into_layout(self) -> $crate::LayoutRef {
                    // SAFETY: LayoutAdapter is #[repr(transparent)] over the Layout type,
                    // guaranteeing identical memory layout and alignment. The transmute is safe
                    // because both types have the same size and representation.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V Layout>], $crate::LayoutAdapter::<$V>>(self) })
                }
            }

            impl From<[<$V Layout>]> for $crate::LayoutRef {
                fn from(value: [<$V Layout>]) -> $crate::LayoutRef {
                    use $crate::IntoLayout;
                    value.into_layout()
                }
            }

            impl AsRef<dyn $crate::LayoutEncoding> for [<$V LayoutEncoding>] {
                fn as_ref(&self) -> &dyn $crate::LayoutEncoding {
                    // SAFETY: LayoutEncodingAdapter is #[repr(transparent)] over the LayoutEncoding type,
                    // which guarantees identical memory layout. This cast is safe because
                    // we're only changing the type metadata, not the actual data.
                    unsafe { &*(self as *const [<$V LayoutEncoding>] as *const $crate::LayoutEncodingAdapter<$V>) }
                }
            }

            impl std::ops::Deref for [<$V LayoutEncoding>] {
                type Target = dyn $crate::LayoutEncoding;

                fn deref(&self) -> &Self::Target {
                    // SAFETY: LayoutEncodingAdapter is #[repr(transparent)] over the LayoutEncoding type,
                    // which guarantees identical memory layout. This cast is safe because
                    // we're only changing the type metadata, not the actual data.
                    unsafe { &*(self as *const [<$V LayoutEncoding>] as *const $crate::LayoutEncodingAdapter<$V>) }
                }
            }
        }
    };
}
