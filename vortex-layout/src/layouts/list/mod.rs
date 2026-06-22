// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
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
/// Child index of the `offsets` layout.
pub const OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the `validity` layout (only present when the list dtype is nullable).
pub const VALIDITY_CHILD_INDEX: usize = 2;

/// Number of children when the list dtype is non-nullable.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 2;

vtable!(List);

impl VTable for List {
    type Layout = ListLayout;
    type Encoding = ListLayoutEncoding;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.list")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ListLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata::new(layout.offsets_ptype()))
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        let mut n = NUM_CHILDREN_NON_NULLABLE;
        if layout.dtype.is_nullable() {
            n += 1;
        }

        n
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match (idx, layout.validity.as_ref()) {
            (ELEMENTS_CHILD_INDEX, _) => Ok(Arc::clone(&layout.elements)),
            (OFFSETS_CHILD_INDEX, _) => Ok(Arc::clone(&layout.offsets)),
            (VALIDITY_CHILD_INDEX, Some(validity)) => Ok(Arc::clone(validity)),
            _ => vortex_bail!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match (idx, layout.validity.is_some()) {
            (ELEMENTS_CHILD_INDEX, _) => LayoutChildType::Auxiliary("elements".into()),
            (OFFSETS_CHILD_INDEX, _) => LayoutChildType::Auxiliary("offsets".into()),
            (VALIDITY_CHILD_INDEX, true) => LayoutChildType::Auxiliary("validity".into()),
            _ => vortex_panic!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ListReader::try_new(
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
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        validate_children(dtype, children.nchildren())?;

        let elements_dtype = dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {dtype}"))?;
        let elements = children.child(ELEMENTS_CHILD_INDEX, elements_dtype.as_ref())?;

        let offsets_dtype = DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable);
        let offsets = children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;

        let validity = dtype
            .is_nullable()
            .then(|| children.child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable)))
            .transpose()?;

        Ok(ListLayout {
            dtype: dtype.clone(),
            elements,
            offsets,
            validity,
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        validate_children(layout.dtype(), children.len())?;

        let mut iter = children.into_iter();
        layout.elements = iter
            .next()
            .ok_or_else(|| vortex_err!("missing elements child"))?;
        layout.offsets = iter
            .next()
            .ok_or_else(|| vortex_err!("missing offsets child"))?;
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

/// Validates expected number of children based on `dtype`
#[inline]
fn validate_children(dtype: &DType, n_children: usize) -> VortexResult<()> {
    let mut expected = NUM_CHILDREN_NON_NULLABLE;

    if dtype.is_nullable() {
        expected += 1;
    };

    vortex_ensure!(
        n_children == expected,
        "ListLayout expects {expected} children, got {n_children}",
    );

    Ok(())
}

#[derive(Debug)]
pub struct ListLayoutEncoding;

/// Stores a list-typed array by shredding `elements`, `offsets`, and optional `validity` children.
#[derive(Clone, Debug)]
pub struct ListLayout {
    dtype: DType,
    elements: LayoutRef,
    offsets: LayoutRef,
    validity: Option<LayoutRef>,
}

impl ListLayout {
    /// Construct a new `ListLayout` from its components.
    ///
    /// # Invariants
    ///
    /// - `dtype` must be a [`DType::List`].
    /// - `validity` must be `Some` iff `dtype.is_nullable()`.
    /// - `offsets.dtype()` must be a non-nullable integer.
    /// - `offsets.row_count()` is the Arrow-canonical `n+1` for `n` lists (or `0` for empty).
    /// - When present, `validity.row_count() == offsets.row_count().saturating_sub(1)`.
    pub fn new(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        Self {
            dtype,
            elements,
            offsets,
            validity,
        }
    }

    /// Number of lists in this layout.
    #[inline]
    pub fn row_count(&self) -> u64 {
        self.offsets.row_count().saturating_sub(1)
    }

    #[inline]
    pub fn elements(&self) -> &LayoutRef {
        &self.elements
    }

    #[inline]
    pub fn offsets(&self) -> &LayoutRef {
        &self.offsets
    }

    #[inline]
    pub fn validity(&self) -> Option<&LayoutRef> {
        self.validity.as_ref()
    }

    /// The integer type used for the `offsets` child layout.
    #[inline]
    pub fn offsets_ptype(&self) -> PType {
        self.offsets.dtype().as_ptype()
    }

    /// The dtype of the inner elements column.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype
            .as_list_element_opt()
            .vortex_expect("ListLayout dtype must be a List")
    }
}

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    offsets_ptype: i32,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_offsets_ptype(offsets_ptype);
        metadata
    }
}
