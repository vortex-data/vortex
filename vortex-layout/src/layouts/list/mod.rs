// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for list-typed columns.

mod expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Child index of the elements layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the offsets layout.
pub const OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the optional validity layout.
pub const VALIDITY_CHILD_INDEX: usize = 2;
/// Number of children for a non-nullable list.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 2;

/// List layout vtable.
#[derive(Clone, Debug)]
pub struct List;

/// Backwards-compatible name for the list layout plugin.
pub use List as ListLayoutEncoding;

/// List-layout-specific data.
#[derive(Clone, Debug)]
pub struct ListData {
    offsets_ptype: PType,
}

/// A list layout shredded into elements, offsets, and optional validity children.
pub type ListLayout = Layout<List>;

impl VTable for List {
    type LayoutData = ListData;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.list");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata::new(layout.offsets_ptype))
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ListLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        ListLayout::validate_children(args.dtype, args.children.nchildren())?;
        let elements_dtype = args
            .dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {}", args.dtype))?;
        args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
        let offsets_dtype = DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable);
        let offsets = args.children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;
        vortex_error::vortex_ensure!(
            offsets.row_count().saturating_sub(1) == args.row_count,
            "List offsets row count does not match parent"
        );
        if args.dtype.is_nullable() {
            let validity = args
                .children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            vortex_error::vortex_ensure!(
                validity.row_count() == args.row_count,
                "List validity row count does not match parent"
            );
        }
        Ok(ListData {
            offsets_ptype: metadata.offsets_ptype(),
        })
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            ELEMENTS_CHILD_INDEX => layout
                .dtype()
                .as_list_element_opt()
                .map(|dtype| dtype.as_ref().clone())
                .ok_or_else(|| vortex_err!("ListLayout requires a List dtype")),
            OFFSETS_CHILD_INDEX => Ok(DType::Primitive(
                layout.offsets_ptype,
                Nullability::NonNullable,
            )),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                Ok(DType::Bool(Nullability::NonNullable))
            }
            _ => vortex_bail!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            ELEMENTS_CHILD_INDEX => LayoutChildType::Auxiliary("elements".into()),
            OFFSETS_CHILD_INDEX => LayoutChildType::Auxiliary("offsets".into()),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                LayoutChildType::Auxiliary("validity".into())
            }
            _ => vortex_panic!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
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
}

impl Layout<List> {
    /// Construct a list layout from its children.
    pub fn new(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> Self {
        let row_count = offsets.row_count().saturating_sub(1);
        let offsets_ptype = offsets.dtype().as_ptype();
        let mut children = vec![elements, offsets];
        children.extend(validity);
        Self::validate_children(&dtype, children.len()).vortex_expect("invalid list children");
        LayoutParts::new(
            List,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            ListData { offsets_ptype },
        )
        .into_typed()
    }

    /// Returns the elements child.
    pub fn elements(&self) -> VortexResult<LayoutRef> {
        self.child(ELEMENTS_CHILD_INDEX)
    }

    /// Returns the offsets child.
    pub fn offsets(&self) -> VortexResult<LayoutRef> {
        self.child(OFFSETS_CHILD_INDEX)
    }

    /// Returns the optional validity child.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.dtype()
            .is_nullable()
            .then(|| self.child(VALIDITY_CHILD_INDEX))
            .transpose()
    }

    /// Returns the integer ptype used by offsets.
    pub fn offsets_ptype(&self) -> PType {
        self.offsets_ptype
    }

    /// Returns the list element dtype.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype()
            .as_list_element_opt()
            .vortex_expect("ListLayout dtype must be a List")
    }

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let expected = NUM_CHILDREN_NON_NULLABLE + usize::from(dtype.is_nullable());
        vortex_ensure_eq!(nchildren, expected);
        Ok(())
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
