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
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_scan::plan::ListIndex;
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
use crate::lowering::slots_from_layout;
use crate::plan::physical::ListPackPlan;
use crate::plan::physical::PlanRef;
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
    /// A coarse copy of the offsets, kept in the layout's metadata: see [`ListIndex`].
    index: Option<ListIndex>,
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
        ProstMetadata(ListLayoutMetadata::new(
            layout.offsets_ptype,
            layout.index.as_ref(),
        ))
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ListLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let offsets_ptype = metadata.get_offsets_ptype()?;
        ListLayout::validate_children(args.dtype, args.children.nchildren())?;
        let elements_dtype = args
            .dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {}", args.dtype))?;
        args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
        let offsets_dtype = DType::Primitive(offsets_ptype, Nullability::NonNullable);
        let offsets = args.children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;
        vortex_ensure_eq!(
            offsets.row_count().saturating_sub(1),
            args.row_count,
            "List offsets row count does not match parent"
        );
        if args.dtype.is_nullable() {
            let validity = args
                .children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            vortex_ensure_eq!(
                validity.row_count(),
                args.row_count,
                "List validity row count does not match parent"
            );
        }
        Ok(ListData {
            offsets_ptype,
            index: metadata.index()?,
        })
    }

    fn nslots(_layout: &Layout<Self>) -> usize {
        // Elements, offsets, and an always-slotted (optionally present) validity child.
        VALIDITY_CHILD_INDEX + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        match slot {
            ELEMENTS_CHILD_INDEX | OFFSETS_CHILD_INDEX => Some(slot),
            VALIDITY_CHILD_INDEX => layout.dtype().is_nullable().then_some(VALIDITY_CHILD_INDEX),
            _ => None,
        }
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

    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        // SAFETY: List layout construction validates its element, offsets, and optional validity
        // child shapes. Its slots are `[elements, offsets, validity]`, which are the ListPack
        // slots. Its index was built from the offsets as they were written.
        let plan = unsafe {
            ListPackPlan::from_slots_unchecked(
                layout.dtype().clone(),
                layout.row_count(),
                slots_from_layout(&layout.to_layout()),
                layout.index.clone(),
            )
        };
        Ok(plan.into_plan())
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
        Self::new_with_index(dtype, elements, offsets, validity, None)
    }

    /// Construct a list layout from its children and an index of its offsets, built from them as
    /// they were written with a [`ListIndexBuilder`](vortex_scan::plan::ListIndexBuilder).
    pub fn new_with_index(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
        index: Option<ListIndex>,
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
            ListData {
                offsets_ptype,
                index,
            },
        )
        .into_typed()
    }

    /// Returns the elements child.
    pub fn elements(&self) -> VortexResult<LayoutRef> {
        self.slot(ELEMENTS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout elements slot is absent"))
    }

    /// Returns the offsets child.
    pub fn offsets(&self) -> VortexResult<LayoutRef> {
        self.slot(OFFSETS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout offsets slot is absent"))
    }

    /// Returns the optional validity child.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.slot(VALIDITY_CHILD_INDEX)
    }

    /// Returns the index of the offsets, if the layout was written with one.
    pub fn index(&self) -> Option<&ListIndex> {
        self.index.as_ref()
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
    #[prost(int32, tag = "1")]
    offsets_ptype: i32,
    /// The index of the offsets: the list each of its windows but the first starts at, as the
    /// difference from the entry before.
    #[prost(uint64, repeated, tag = "2")]
    index_lists: Vec<u64>,
    /// The element row each of those lists starts at, as the difference from the entry before.
    #[prost(uint64, repeated, tag = "3")]
    index_offsets: Vec<u64>,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType, index: Option<&ListIndex>) -> Self {
        let (index_lists, index_offsets) = match index {
            Some(index) => (deltas(index.lists()), deltas(index.offsets())),
            None => (Vec::new(), Vec::new()),
        };
        Self {
            offsets_ptype: offsets_ptype as i32,
            index_lists,
            index_offsets,
        }
    }

    /// The index of the offsets, if the layout was written with one.
    fn index(&self) -> VortexResult<Option<ListIndex>> {
        if self.index_lists.is_empty() && self.index_offsets.is_empty() {
            return Ok(None);
        }
        Ok(Some(ListIndex::try_new(
            prefix_sums(&self.index_lists)?,
            prefix_sums(&self.index_offsets)?,
        )?))
    }

    fn get_offsets_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.offsets_ptype).map_err(|_| {
            vortex_err!(
                "ListLayoutMetadata offsets_ptype must be a known PType, got {}",
                self.offsets_ptype
            )
        })
    }
}

/// Each value of increasing `values` as its difference from the one before, the first from
/// zero: small numbers, which the metadata stores in few bytes.
fn deltas(values: &[u64]) -> Vec<u64> {
    let mut before = 0;
    values
        .iter()
        .map(|&value| {
            let delta = value - before;
            before = value;
            delta
        })
        .collect()
}

/// The values whose differences are `deltas`.
fn prefix_sums(deltas: &[u64]) -> VortexResult<Buffer<u64>> {
    let mut value = 0u64;
    deltas
        .iter()
        .map(|&delta| {
            value = value
                .checked_add(delta)
                .ok_or_else(|| vortex_err!("ListLayoutMetadata index overflows"))?;
            Ok(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_scan::plan::ListIndex;
    use vortex_session::registry::ReadContext;

    use super::List;
    use super::ListLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::layout_children;
    use crate::test::SESSION;

    /// An index survives the metadata, which stores it as differences, and a layout without one
    /// stores nothing.
    #[test]
    fn metadata_round_trips_the_index() -> VortexResult<()> {
        let index = ListIndex::try_new(
            Buffer::from_iter([3u64, 4, 900]),
            Buffer::from_iter([7u64, 7, 1 << 40]),
        )?;
        let encoded = ListLayoutMetadata::new(PType::U64, Some(&index)).encode_to_vec();
        let decoded = ListLayoutMetadata::decode(encoded.as_slice())
            .map_err(|err| vortex_err!("metadata does not decode: {err}"))?;
        assert_eq!(decoded.index()?, Some(index));

        let without = ListLayoutMetadata::new(PType::U64, None);
        assert_eq!(without.index()?, None);
        assert_eq!(
            without.encode_to_vec(),
            ListLayoutMetadata {
                offsets_ptype: PType::U64 as i32,
                index_lists: Vec::new(),
                index_offsets: Vec::new(),
            }
            .encode_to_vec()
        );
        Ok(())
    }

    #[test]
    fn test_build_rejects_unknown_offsets_ptype() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let metadata = ListLayoutMetadata {
            offsets_ptype: 11,
            index_lists: Vec::new(),
            index_offsets: Vec::new(),
        }
        .encode_to_vec();
        let children = layout_children(vec![]);
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &SESSION,
            array_read_ctx: &read_ctx,
        };
        let result = LayoutPlugin::deserialize(
            &List,
            LayoutDeserialization {
                serialized_id: LayoutPlugin::id(&List),
                dtype: &dtype,
                row_count: 0,
                metadata: &metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &build_ctx,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("ListLayoutMetadata offsets_ptype must be a known PType, got 11")
        }));

        assert_eq!(
            ListLayoutMetadata::new(PType::U32, None).get_offsets_ptype()?,
            PType::U32
        );

        Ok(())
    }
}
