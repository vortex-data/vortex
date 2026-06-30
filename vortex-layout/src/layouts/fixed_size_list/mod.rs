// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::FixedSizeListReader;
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

/// Child index of the `elements` layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the `validity` layout, only present when the fixed-size list dtype is nullable.
pub const VALIDITY_CHILD_INDEX: usize = 1;

vtable!(FixedSizeList);

impl VTable for FixedSizeList {
    type Layout = FixedSizeListLayout;
    type Encoding = FixedSizeListLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.fixed_size_list")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(FixedSizeListLayoutEncoding.as_ref())
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
        1 + usize::from(layout.dtype.is_nullable())
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match (idx, layout.validity.as_ref()) {
            (ELEMENTS_CHILD_INDEX, _) => Ok(Arc::clone(&layout.elements)),
            (VALIDITY_CHILD_INDEX, Some(validity)) => Ok(Arc::clone(validity)),
            _ => vortex_bail!("Invalid child index {idx} for FixedSizeListLayout"),
        }
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match (idx, layout.validity.is_some()) {
            (ELEMENTS_CHILD_INDEX, _) => LayoutChildType::Auxiliary("elements".into()),
            (VALIDITY_CHILD_INDEX, true) => LayoutChildType::Auxiliary("validity".into()),
            _ => vortex_panic!("Invalid child index {idx} for FixedSizeListLayout"),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FixedSizeListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx,
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        validate_children(dtype, row_count, children)?;

        let element_dtype = dtype
            .as_fixed_size_list_element_opt()
            .ok_or_else(|| vortex_err!("FixedSizeListLayout requires a FixedSizeList dtype"))?;
        let elements = children.child(ELEMENTS_CHILD_INDEX, element_dtype)?;
        let validity = dtype
            .is_nullable()
            .then(|| children.child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable)))
            .transpose()?;

        Ok(FixedSizeListLayout {
            row_count,
            dtype: dtype.clone(),
            elements,
            validity,
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        validate_child_count(layout.dtype(), children.len())?;

        let mut iter = children.into_iter();
        layout.elements = iter
            .next()
            .ok_or_else(|| vortex_err!("missing elements child"))?;
        layout.validity = layout
            .dtype
            .is_nullable()
            .then(|| {
                iter.next()
                    .ok_or_else(|| vortex_err!("missing validity child"))
            })
            .transpose()?;
        Ok(())
    }
}

fn validate_child_count(dtype: &DType, nchildren: usize) -> VortexResult<()> {
    let expected = 1 + usize::from(dtype.is_nullable());
    vortex_ensure!(
        nchildren == expected,
        "FixedSizeListLayout expects {expected} children, got {nchildren}"
    );
    Ok(())
}

fn validate_children(
    dtype: &DType,
    row_count: u64,
    children: &dyn LayoutChildren,
) -> VortexResult<()> {
    validate_child_count(dtype, children.nchildren())?;
    let DType::FixedSizeList(_, list_size, _) = dtype else {
        vortex_bail!("FixedSizeListLayout requires a FixedSizeList dtype, got {dtype}");
    };
    let expected_elements = row_count
        .checked_mul(u64::from(*list_size))
        .ok_or_else(|| vortex_err!("fixed-size list elements row count overflow"))?;
    let actual_elements = children.child_row_count(ELEMENTS_CHILD_INDEX);
    vortex_ensure!(
        actual_elements == expected_elements,
        "FixedSizeListLayout elements row count {actual_elements} does not match expected {expected_elements}"
    );
    if dtype.is_nullable() {
        let validity_rows = children.child_row_count(VALIDITY_CHILD_INDEX);
        vortex_ensure!(
            validity_rows == row_count,
            "FixedSizeListLayout validity row count {validity_rows} does not match row count {row_count}"
        );
    }
    Ok(())
}

#[derive(Debug)]
pub struct FixedSizeListLayoutEncoding;

/// Stores a fixed-size list by shredding elements and optional list validity into child layouts.
#[derive(Clone, Debug)]
pub struct FixedSizeListLayout {
    row_count: u64,
    dtype: DType,
    elements: LayoutRef,
    validity: Option<LayoutRef>,
}

impl FixedSizeListLayout {
    /// Construct a fixed-size-list layout from its components.
    ///
    /// # Invariants
    ///
    /// - `dtype` must be a [`DType::FixedSizeList`].
    /// - `elements.row_count() == row_count * list_size`.
    /// - `validity` is present iff `dtype.is_nullable()`.
    pub fn new(
        row_count: u64,
        dtype: DType,
        elements: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        Self {
            row_count,
            dtype,
            elements,
            validity,
        }
    }

    /// Number of fixed-size-list rows in this layout.
    #[inline]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    #[inline]
    pub fn elements(&self) -> &LayoutRef {
        &self.elements
    }

    #[inline]
    pub fn validity(&self) -> Option<&LayoutRef> {
        self.validity.as_ref()
    }

    /// The fixed number of elements in each list row.
    #[inline]
    pub fn list_size(&self) -> u32 {
        match &self.dtype {
            DType::FixedSizeList(_, list_size, _) => *list_size,
            _ => vortex_panic!("FixedSizeListLayout dtype must be FixedSizeList"),
        }
    }

    /// The dtype of the inner elements column.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype
            .as_fixed_size_list_element_opt()
            .vortex_expect("FixedSizeListLayout dtype must be FixedSizeList")
    }
}
