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
use crate::lowering::slots_from_layout;
use crate::plan::physical::ElementSamples;
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
    element_samples: Option<ElementSamples>,
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
        let mut metadata = ListLayoutMetadata::new(layout.offsets_ptype);
        if let Some(samples) = &layout.element_samples {
            metadata.set_element_samples(samples);
        }
        ProstMetadata(metadata)
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
        let elements = args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
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
        let element_samples = metadata.get_element_samples(args.row_count, elements.row_count())?;
        Ok(ListData {
            offsets_ptype,
            element_samples,
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
        // slots.
        let plan = unsafe {
            ListPackPlan::from_slots_unchecked(
                layout.dtype().clone(),
                layout.row_count(),
                slots_from_layout(&layout.to_layout()),
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
        Self::new_with_element_samples(dtype, elements, offsets, validity, None)
    }

    /// Construct a list layout from its children, with the writer's samples of the list holding
    /// every so many elements.
    pub fn new_with_element_samples(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
        element_samples: Option<ElementSamples>,
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
                element_samples,
            },
        )
        .into_typed()
    }

    /// The list holding every so many elements, when the writer recorded them.
    pub fn element_samples(&self) -> Option<&ElementSamples> {
        self.element_samples.as_ref()
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
    /// The element rows between [`ElementSamples`], or zero when the writer recorded none.
    #[prost(uint64, tag = "2")]
    element_stride: u64,
    /// The list holding every `element_stride`-th element row, delta-encoded: the first sampled
    /// list, then the difference between each sampled list and the one before it. The sampled
    /// lists never decrease, so the deltas are small varints.
    #[prost(uint64, repeated, packed = "true", tag = "3")]
    element_list_deltas: Vec<u64>,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType) -> Self {
        Self {
            offsets_ptype: offsets_ptype as i32,
            element_stride: 0,
            element_list_deltas: Vec::new(),
        }
    }

    fn set_element_samples(&mut self, samples: &ElementSamples) {
        self.element_stride = samples.stride();
        // The samples never decrease, so the subtractions do not wrap.
        self.element_list_deltas = samples
            .lists()
            .iter()
            .scan(0u64, |previous, &list| {
                let delta = list - *previous;
                *previous = list;
                Some(delta)
            })
            .collect();
    }

    /// The writer's [`ElementSamples`] of a layout of `list_rows` lists of `element_rows`
    /// elements, or `None` when it recorded none.
    fn get_element_samples(
        &self,
        list_rows: u64,
        element_rows: u64,
    ) -> VortexResult<Option<ElementSamples>> {
        if self.element_stride == 0 {
            return Ok(None);
        }
        let mut list = 0u64;
        let lists = self
            .element_list_deltas
            .iter()
            .map(|&delta| {
                list = list.checked_add(delta).ok_or_else(|| {
                    vortex_err!("ListLayoutMetadata element samples overflow a list index")
                })?;
                Ok(list)
            })
            .collect::<VortexResult<Vec<u64>>>()?;
        ElementSamples::try_new(self.element_stride, lists, list_rows, element_rows).map(Some)
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::List;
    use super::ListLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::layout_children;
    use crate::plan::physical::ElementSamples;
    use crate::test::SESSION;

    #[test]
    fn test_build_rejects_unknown_offsets_ptype() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let metadata = ListLayoutMetadata {
            offsets_ptype: 11,
            ..Default::default()
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
            ListLayoutMetadata::new(PType::U32).get_offsets_ptype()?,
            PType::U32
        );

        Ok(())
    }

    /// The sampled lists are stored as the first list and then the differences between
    /// consecutive lists, and decode back to the same samples.
    #[test]
    fn element_samples_round_trip_as_deltas() -> VortexResult<()> {
        let samples = ElementSamples::try_new(4, vec![0, 0, 3, 3, 9, 1_000], 1_001, 24)?;
        let mut metadata = ListLayoutMetadata::new(PType::U64);
        metadata.set_element_samples(&samples);
        assert_eq!(metadata.element_stride, 4);
        assert_eq!(metadata.element_list_deltas, [0, 0, 3, 0, 6, 991]);

        let decoded = ListLayoutMetadata::decode(metadata.encode_to_vec().as_slice())?;
        assert_eq!(decoded.get_element_samples(1_001, 24)?, Some(samples));
        Ok(())
    }

    /// Metadata written before element samples, or without them, reads as having none.
    #[test]
    fn metadata_without_element_samples_reads_as_none() -> VortexResult<()> {
        let bytes = ListLayoutMetadata::new(PType::U32).encode_to_vec();
        let decoded = ListLayoutMetadata::decode(bytes.as_slice())?;
        assert_eq!(decoded.get_offsets_ptype()?, PType::U32);
        assert_eq!(decoded.get_element_samples(10, 100)?, None);
        Ok(())
    }

    #[rstest]
    #[case::too_few_samples(2, vec![0, 1], 3, 5, "cover every stride")]
    #[case::too_many_samples(2, vec![0, 1, 0, 0], 3, 5, "cover every stride")]
    #[case::overflowing_list(1, vec![u64::MAX, 1], 2, 2, "overflow")]
    #[case::list_out_of_range(1, vec![0, 3], 3, 2, "is not one of the 3 lists")]
    fn corrupt_element_samples_are_rejected(
        #[case] element_stride: u64,
        #[case] element_list_deltas: Vec<u64>,
        #[case] list_rows: u64,
        #[case] element_rows: u64,
        #[case] message: &str,
    ) {
        let metadata = ListLayoutMetadata {
            offsets_ptype: PType::U64 as i32,
            element_stride,
            element_list_deltas,
        };
        let err = metadata
            .get_element_samples(list_rows, element_rows)
            .expect_err("corrupt samples must be rejected");
        assert!(err.to_string().contains(message), "{err}");
    }
}
