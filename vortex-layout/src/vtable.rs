use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use vortex_array::{ArrayContext, DeserializeMetadata, SerializeMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;

use crate::layout::LayoutRef;
use crate::segments::{SegmentId, SegmentSource};
use crate::visitor::ReaderVisitor;
use crate::{LayoutId, LayoutReader, ReaderChildren};

pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Reader: 'static + Send + Sync + Deref<Target = dyn LayoutReader>;
    type Layout: 'static + Send + Sync;
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    /// Returns the ID of the layout.
    fn id(layout: &Self::Layout) -> LayoutId;

    /// Returns the layout for the layout reader.
    fn layout(reader: &Self::Reader) -> LayoutRef;

    /// Returns the row count for the layout reader.
    fn row_count(reader: &Self::Reader) -> u64;

    /// Returns the dtype for the layout reader.
    fn dtype(reader: &Self::Reader) -> DType;

    /// Visitor the children of the layout reader.
    fn visit_children(
        reader: &Self::Reader,
        field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn ReaderVisitor,
    );

    /// Construct a new [`LayoutReader`] from the provided parts.
    fn reader_from_parts(
        layout: &Self::Layout,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn ReaderChildren,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Self::Reader>;
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

            impl $crate::IntoLayout for [<$V Layout>] {
                fn into_layout(self) -> $crate::LayoutRef {
                    // We can unsafe transmute ourselves to an LayoutAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V Layout>], $crate::LayoutAdapter::<[<$V VTable>]>>(self) })
                }
            }

            impl AsRef<dyn $crate::LayoutReader> for [<$V Reader>] {
                fn as_ref(&self) -> &dyn $crate::LayoutReader {
                    // We can unsafe cast ourselves to an LayoutReaderAdapter.
                    unsafe { &*(self as *const [<$V Reader>] as *const $crate::LayoutReaderAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Reader>] {
                type Target = dyn $crate::LayoutReader;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an LayoutReaderAdapter.
                    unsafe { &*(self as *const [<$V Reader>] as *const $crate::LayoutReaderAdapter<[<$V VTable>]>) }
                }
            }
        }
    };
}
