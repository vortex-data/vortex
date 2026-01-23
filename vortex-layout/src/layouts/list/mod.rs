// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::ArrayContext;
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(List);

impl VTable for ListVTable {
    type Layout = ListLayout;
    type Encoding = ListLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.list")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ListLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        let validity_children = layout.dtype.is_nullable() as usize;
        match &layout.dtype {
            DType::List(..) => 2 + validity_children, // offsets + elements
            DType::FixedSizeList(..) => 1 + validity_children, // elements
            _ => 0,
        }
    }

    fn child(layout: &Self::Layout, index: usize) -> VortexResult<LayoutRef> {
        let is_nullable = layout.dtype.is_nullable();
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        let child_dtype = match (&layout.dtype, is_nullable, index) {
            // validity
            (_, true, 0) => DType::Bool(Nullability::NonNullable),

            // variable-size list
            (DType::List(..), false, 0) => offsets_dtype,
            (DType::List(element_dtype, _), false, 1) => (*element_dtype.as_ref()).clone(),
            (DType::List(..), true, 1) => offsets_dtype,
            (DType::List(element_dtype, _), true, 2) => (*element_dtype.as_ref()).clone(),

            // fixed-size list
            (DType::FixedSizeList(element_dtype, ..), false, 0) => {
                (*element_dtype.as_ref()).clone()
            }
            (DType::FixedSizeList(element_dtype, ..), true, 1) => (*element_dtype.as_ref()).clone(),

            _ => return Err(vortex_err!("Invalid child index {index} for list layout")),
        };

        layout.children.child(index, &child_dtype)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        let is_nullable = layout.dtype.is_nullable();

        if is_nullable && idx == 0 {
            return LayoutChildType::Auxiliary("validity".into());
        }

        match &layout.dtype {
            DType::List(..) => {
                let offsets_idx = if is_nullable { 1 } else { 0 };
                if idx == offsets_idx {
                    LayoutChildType::Auxiliary("offsets".into())
                } else {
                    LayoutChildType::Auxiliary("elements".into())
                }
            }
            DType::FixedSizeList(..) => LayoutChildType::Auxiliary("elements".into()),
            _ => unreachable!(
                "ListLayout only supports List and FixedSizeList dtypes, got {}",
                layout.dtype()
            ),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ArrayContext,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure!(
            matches!(dtype, DType::List(..) | DType::FixedSizeList(..)),
            "Expected list dtype, got {}",
            dtype
        );

        let expected_children = match dtype {
            DType::List(..) => 2 + (dtype.is_nullable() as usize),
            DType::FixedSizeList(..) => 1 + (dtype.is_nullable() as usize),
            _ => unreachable!(),
        };
        vortex_ensure!(
            children.nchildren() == expected_children,
            "List layout has {} children, expected {}",
            children.nchildren(),
            expected_children
        );

        Ok(ListLayout {
            row_count,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        let expected_children = match layout.dtype {
            DType::List(..) => 2 + (layout.dtype.is_nullable() as usize),
            DType::FixedSizeList(..) => 1 + (layout.dtype.is_nullable() as usize),
            _ => vortex_bail!("Expected list dtype, got {}", layout.dtype),
        };
        vortex_ensure!(
            children.len() == expected_children,
            "ListLayout expects {} children, got {}",
            expected_children,
            children.len()
        );
        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }
}

#[derive(Debug)]
pub struct ListLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ListLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl ListLayout {
    pub fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        Self {
            row_count,
            dtype,
            children: OwnedLayoutChildren::layout_children(children),
        }
    }

    #[inline]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }
}
