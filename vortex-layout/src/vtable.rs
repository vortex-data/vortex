use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use vortex_array::{ArrayContext, DeserializeMetadata, SerializeMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    Layout, LayoutChildren, LayoutEncoding, LayoutEncodingRef, LayoutId, LayoutReaderRef,
    LayoutVisitor,
};

pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Layout: 'static + Send + Sync + Deref<Target = dyn Layout>;
    type Encoding: 'static + Send + Sync + Deref<Target = dyn LayoutEncoding>;
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    /// Returns the ID of the layout encoding.
    fn id(encoding: &Self::Encoding) -> LayoutId;

    /// Returns the encoding for the layout.
    fn encoding(layout: &Self::Layout) -> LayoutEncodingRef;

    /// Returns the row count for the layout reader.
    fn row_count(layout: &Self::Layout) -> u64;

    /// Returns the dtype for the layout reader.
    fn dtype(layout: &Self::Layout) -> &DType;

    /// Returns the number of children for the layout.
    fn nchildren(layout: &Self::Layout) -> usize;

    /// Visitor the children of the layout.
    fn visit_children(
        layout: &Self::Layout,
        field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    );

    /// Returns the segment IDs for the layout.
    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId>;

    /// Create a new reader for the layout.
    fn new_reader(
        layout: &Arc<Self::Layout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef>;

    /// Construct a new [`Layout`] from the provided parts.
    fn build(
        encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout>;
}

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct [<$V VTable>];

            impl AsRef<dyn $crate::Layout> for [<$V Layout>] {
                fn as_ref(&self) -> &dyn $crate::Layout {
                    // We can unsafe cast ourselves to a LayoutAdapter.
                    unsafe { &*(self as *const [<$V Layout>] as *const $crate::LayoutAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Layout>] {
                type Target = dyn $crate::Layout;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an LayoutAdapter.
                    unsafe { &*(self as *const [<$V Layout>] as *const $crate::LayoutAdapter<[<$V VTable>]>) }
                }
            }

            impl AsRef<dyn $crate::LayoutEncoding> for [<$V LayoutEncoding>] {
                fn as_ref(&self) -> &dyn $crate::LayoutEncoding {
                    // We can unsafe cast ourselves to an LayoutEncodingAdapter.
                    unsafe { &*(self as *const [<$V LayoutEncoding>] as *const $crate::LayoutEncodingAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V LayoutEncoding>] {
                type Target = dyn $crate::LayoutEncoding;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an LayoutEncodingAdapter.
                    unsafe { &*(self as *const [<$V LayoutEncoding>] as *const $crate::LayoutEncodingAdapter<[<$V VTable>]>) }
                }
            }
        }
    };
}
