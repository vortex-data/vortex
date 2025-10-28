// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod reader;
pub mod writer;

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

/// Variants of lists writable as a `vortex.list` layout.
#[derive(Clone, Debug)]
pub enum ListLayoutInner {
    List {
        validity: Option<LayoutRef>,
        offsets: LayoutRef,
        elements: LayoutRef,
    },
    FixedSizeList {
        stride: u32,
        validity: Option<LayoutRef>,
        elements: LayoutRef,
    },
}

#[derive(Clone, Debug)]
pub struct ListLayout {
    /// Type of the array contained in the ListLayout.
    ///
    /// This should be one of `List` or `FixedSizeList`.
    pub dtype: DType,
    /// Count of rows represented in the ListLayout.
    ///
    /// This is required because `FixedSizeList` supports a stride of 0, which means that we cannot
    /// infer its total length by simple division.
    pub row_count: u64,
    /// Total number of elements across all lists in the layout.
    ///
    /// This is different from the row count, which is the number of other element types that we have
    pub elements_count: u64,
    /// The inner layout children and metadata, varies based on if the type is
    /// `List` or `FixedSizeList`.
    pub inner: Arc<ListLayoutInner>,
}

use vortex_dtype::PType;
use vortex_session::VortexSession;

use crate::layouts::list::reader::ListReader;

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(uint64, tag = "1")]
    pub row_count: u64,
    #[prost(optional, uint32, tag = "2")]
    pub fixed_size_stride: Option<u32>,
    #[prost(optional, enumeration = "PType", tag = "3")]
    pub offsets_ptype: Option<i32>,
}

impl ListLayoutMetadata {
    pub fn new_list(row_count: u64, offsets_ptype: PType) -> Self {
        let mut this = Self {
            row_count,
            ..Default::default()
        };
        this.set_offsets_ptype(offsets_ptype);
        this
    }

    pub fn new_fixed_size_list(row_count: u64, stride: u32) -> Self {
        Self {
            row_count,
            fixed_size_stride: Some(stride),
            offsets_ptype: None,
        }
    }
}

vtable!(List);

#[derive(Debug)]
pub struct ListLayoutEncoding;

impl VTable for ListVTable {
    type Layout = ListLayout;
    type Encoding = ListLayoutEncoding;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.list")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        todo!()
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        match &*layout.inner {
            ListLayoutInner::List { offsets, .. } => {
                // Store based on the row count instead
                ProstMetadata(ListLayoutMetadata::new_list(
                    layout.row_count,
                    offsets.dtype().as_ptype(),
                ))
            }
            ListLayoutInner::FixedSizeList { stride, .. } => ProstMetadata(
                ListLayoutMetadata::new_fixed_size_list(layout.row_count, *stride),
            ),
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        Vec::new()
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        match &*layout.inner {
            ListLayoutInner::List {
                validity,
                offsets,
                elements,
            } => {
                offsets.nchildren()
                    + elements.nchildren()
                    + validity.as_ref().map(|l| l.nchildren()).unwrap_or_default()
            }
            ListLayoutInner::FixedSizeList {
                validity, elements, ..
            } => {
                elements.nchildren() + validity.as_ref().map(|l| l.nchildren()).unwrap_or_default()
            }
        }
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match &*layout.inner {
            ListLayoutInner::List {
                validity,
                offsets,
                elements,
            } => match validity {
                None => match idx {
                    0 => Ok(offsets.clone()),
                    1 => Ok(elements.clone()),
                    _ => vortex_bail!("Invalid index {idx} for ListLayout::List with 2 children"),
                },
                Some(validity) => match idx {
                    0 => Ok(validity.clone()),
                    1 => Ok(offsets.clone()),
                    2 => Ok(elements.clone()),
                    _ => vortex_bail!("Invalid index {idx} for ListLayout::List with 3 children"),
                },
            },
            ListLayoutInner::FixedSizeList {
                validity, elements, ..
            } => match validity {
                None => match idx {
                    0 => Ok(elements.clone()),
                    _ => vortex_bail!(
                        "Invalid index {idx} for ListLayout::FixedSizeList with 1 child"
                    ),
                },
                Some(validity) => match idx {
                    0 => Ok(validity.clone()),
                    1 => Ok(elements.clone()),
                    _ => vortex_bail!(
                        "Invalid index {idx} for ListLayout::FixedSizeList with 2 children"
                    ),
                },
            },
        }
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match &*layout.inner {
            ListLayoutInner::List { validity, .. } => match validity {
                None => match idx {
                    0 => LayoutChildType::Auxiliary(Arc::from("offsets")),
                    1 => LayoutChildType::Auxiliary(Arc::from("elements")),
                    _ => vortex_panic!("Invalid index {idx} for ListLayout::List with 2 children"),
                },
                Some(_) => match idx {
                    0 => LayoutChildType::Auxiliary(Arc::from("validity")),
                    1 => LayoutChildType::Auxiliary(Arc::from("offsets")),
                    2 => LayoutChildType::Auxiliary(Arc::from("elements")),
                    _ => vortex_panic!("Invalid index {idx} for ListLayout::List with 3 children"),
                },
            },
            ListLayoutInner::FixedSizeList { validity, .. } => match validity {
                None => match idx {
                    0 => LayoutChildType::Auxiliary(Arc::from("elements")),
                    _ => vortex_panic!(
                        "Invalid index {idx} for ListLayout::FixedSizeList with 1 child"
                    ),
                },
                Some(_) => match idx {
                    0 => LayoutChildType::Auxiliary(Arc::from("validity")),
                    1 => LayoutChildType::Auxiliary(Arc::from("elements")),
                    _ => vortex_panic!(
                        "Invalid index {idx} for ListLayout::FixedSizeList with 2 children"
                    ),
                },
            },
        }
    }

    fn new_reader(
        layout: &ListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        match &*layout.inner {
            l @ ListLayoutInner::List {
                offsets,
                elements,
                validity,
            } => {
                let offsets_name: Arc<str> = format!("{name}.offsets").as_str().into();
                let elements_name: Arc<str> = format!("{name}.elements").as_str().into();

                let offsets_reader =
                    offsets.new_reader(offsets_name, segment_source.clone(), session)?;

                let elements_reader =
                    elements.new_reader(elements_name, segment_source.clone(), session)?;

                let validity_reader = validity
                    .as_ref()
                    .map(|v| {
                        let validity_name: Arc<str> = format!("{name}.validity").as_str().into();

                        v.new_reader(validity_name, segment_source.clone(), session)
                    })
                    .transpose()?;

                Ok(Arc::new(ListReader::new(
                    name,
                    layout.clone(),
                    offsets_reader,
                    elements_reader,
                    validity_reader,
                )))
            }
            ListLayoutInner::FixedSizeList { .. } => {
                todo!()
            }
        }
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure!(
            segment_ids.is_empty(),
            "ListLayout does not hold segments, cannot build with {} segments",
            segment_ids.len()
        );

        match dtype {
            // We don't know what this means. We should just write the FixedSize elements together,
            // and just make sure that the elements are always coalesced in units of XYZ size.
            DType::List(elem_type, nullability) => {
                let (validity, next_child) = if nullability.is_nullable() {
                    (
                        Some(children.child(0, &DType::Bool(Nullability::NonNullable))?),
                        1,
                    )
                } else {
                    (None, 0)
                };

                let offsets = children.child(next_child, &metadata.offsets_ptype().into())?;
                let elements = children.child(next_child + 1, elem_type)?;

                // Verify that the row count matches the expected row count based on the
                // length of the offsets.
                let num_offsets = offsets.row_count();
                vortex_ensure!(
                    (num_offsets == 0 && row_count == 0) || (num_offsets - 1 == row_count),
                    "ListLayout: row_count {row_count} mismatch for List with offsets child of length {num_offsets}"
                );

                Ok(ListLayout {
                    row_count,
                    dtype: dtype.clone(),
                    elements_count: elements.row_count(),
                    inner: Arc::new(ListLayoutInner::List {
                        validity,
                        offsets,
                        elements,
                    }),
                })
            }
            DType::FixedSizeList(elem_type, stride, nullability) => {
                let (validity, next_child) = if nullability.is_nullable() {
                    (
                        Some(children.child(0, &DType::Bool(Nullability::NonNullable))?),
                        1,
                    )
                } else {
                    (None, 0)
                };

                let elements = children.child(next_child, elem_type)?;

                if *stride > 0 {
                    // Common case: stride is non-zero, then the number of elements should be
                    // row_count * stride.
                    let stride = *stride as u64;
                    let num_elems = elements.row_count();
                    vortex_ensure!(
                        num_elems / stride == row_count,
                        "ListLayout: row count {row_count} mismatch for FixedSizeList with stride {stride} and {num_elems} elements"
                    );
                }

                Ok(ListLayout {
                    row_count,
                    dtype: dtype.clone(),
                    elements_count: elements.row_count(),
                    inner: Arc::new(ListLayoutInner::FixedSizeList {
                        stride: *stride,
                        validity,
                        elements,
                    }),
                })
            }
            _ => vortex_bail!("Cannot build ListLayout to read data with type {dtype}"),
        }
    }
}
