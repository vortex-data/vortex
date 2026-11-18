// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for list-typed columns.

mod expr;
mod reader;
pub mod writer;

use std::ops::Range;
use std::sync::Arc;

use reader::ListReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
use crate::plan::physical::ListPackPlan;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
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

/// The list holding every `stride`-th element row, recorded by the writer.
///
/// A reader maps a selection of lists to the element rows that hold them, to within `stride`
/// elements, without reading the offsets: list `i` holds element row `k * stride` exactly when
/// it is the last list starting at or before that row. Sampling the elements rather than the
/// lists bounds what a selection reads in elements, whatever the lists' lengths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementSamples {
    stride: u64,
    lists: Arc<[u64]>,
}

impl ElementSamples {
    /// Samples every `stride`-th of `element_rows` elements of `list_rows` lists: one list per
    /// sample, never decreasing.
    pub fn try_new(
        stride: u64,
        lists: Vec<u64>,
        list_rows: u64,
        element_rows: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(stride > 0, "List element samples need a stride");
        vortex_ensure_eq!(
            lists.len() as u64,
            element_rows.div_ceil(stride),
            "List element samples do not cover the elements"
        );
        vortex_ensure!(
            lists.windows(2).all(|pair| pair[0] <= pair[1])
                && lists.last().is_none_or(|&last| last < list_rows),
            "List element samples must be ascending lists of the layout"
        );
        Ok(Self {
            stride,
            lists: lists.into(),
        })
    }

    /// The elements between samples.
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// The list holding element row `k * stride`, for each `k`.
    pub fn lists(&self) -> &[u64] {
        &self.lists
    }

    /// The element rows holding `lists` in a layout of `element_rows` elements: every element of
    /// those lists, and at most `stride` more on either side.
    pub fn element_range(&self, lists: Range<u64>, element_rows: u64) -> Range<u64> {
        // Sample `k` is at or before the first list's first element exactly when an earlier
        // list holds it, and before the last list's end exactly when a list before the end
        // holds it.
        let after_start = self.lists.partition_point(|&list| list < lists.start) as u64;
        let through_end = self.lists.partition_point(|&list| list < lists.end) as u64;
        if through_end == 0 {
            // No list before the end holds an element.
            return 0..0;
        }
        after_start.saturating_sub(1) * self.stride..(through_end * self.stride).min(element_rows)
    }
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
            metadata.element_stride = samples.stride;
            metadata.element_lists = samples.lists.to_vec();
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
        let element_samples = (metadata.element_stride > 0)
            .then(|| {
                ElementSamples::try_new(
                    metadata.element_stride,
                    metadata.element_lists.clone(),
                    args.row_count,
                    elements.row_count(),
                )
            })
            .transpose()?;
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
                PlanSlots::from_layout(&layout.to_layout()),
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
    /// The elements between [`ElementSamples`], or zero when the writer recorded none.
    #[prost(uint64, tag = "2")]
    element_stride: u64,
    /// The list holding every `element_stride`-th element row.
    #[prost(uint64, repeated, tag = "3")]
    element_lists: Vec<u64>,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType) -> Self {
        Self {
            offsets_ptype: offsets_ptype as i32,
            element_stride: 0,
            element_lists: Vec::new(),
        }
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::ElementSamples;
    use super::List;
    use super::ListLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::layout_children;
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

    /// Every run of lists maps to element rows holding all of their elements, and at most a
    /// stride more on either side.
    #[test]
    fn element_range_bounds_the_elements_of_any_lists() -> VortexResult<()> {
        let lengths = [0u64, 0, 5, 1, 0, 40, 3, 0, 0, 7, 1, 1, 0, 64, 2, 0];
        let mut offsets = vec![0u64];
        for length in lengths {
            offsets.push(offsets[offsets.len() - 1] + length);
        }
        let element_rows = offsets[offsets.len() - 1];
        let list_rows = lengths.len() as u64;
        for stride in [1, 3, 8, 16, 1_000] {
            // The list holding each sampled element: the last list starting at or before it.
            let lists = (0..element_rows)
                .step_by(usize::try_from(stride)?)
                .map(|row| {
                    offsets[..lengths.len()].partition_point(|&start| start <= row) as u64 - 1
                })
                .collect();
            let samples = ElementSamples::try_new(stride, lists, list_rows, element_rows)?;
            for start in 0..list_rows {
                for end in start..=list_rows {
                    let (first, last) = (
                        offsets[usize::try_from(start)?],
                        offsets[usize::try_from(end)?],
                    );
                    let range = samples.element_range(start..end, element_rows);
                    assert!(
                        range.start <= first && last <= range.end,
                        "{start}..{end} @ {stride}"
                    );
                    if first < last {
                        assert!(first <= range.start + stride && range.end < last + stride);
                    }
                }
            }
        }
        Ok(())
    }
}
