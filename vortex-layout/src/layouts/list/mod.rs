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
/// Child index of the validity layout, present exactly when the list dtype is nullable.
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
        let offsets_ptype = metadata.get_offsets_ptype()?;
        let elements_dtype = args
            .dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {}", args.dtype))?;
        // Trailing children are reserved for ones that readers may ignore; see `ListLayoutMetadata`.
        let required_children = ListLayout::required_children(args.dtype);
        vortex_ensure!(
            args.children.nchildren() >= required_children,
            "ListLayout of dtype {} expects at least {required_children} children, got {}",
            args.dtype,
            args.children.nchildren()
        );
        args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
        let offsets_dtype = DType::Primitive(offsets_ptype, Nullability::NonNullable);
        let offsets = args.children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;
        let offsets_row_count = args.row_count.checked_add(1).ok_or_else(|| {
            vortex_err!(
                "ListLayout row count {} overflows its offsets row count",
                args.row_count
            )
        })?;
        vortex_ensure_eq!(
            offsets.row_count(),
            offsets_row_count,
            "List offsets must have one more row than the list"
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
        Ok(ListData { offsets_ptype })
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

    /// The number of children the wire contract defines for a list of `dtype`.
    fn required_children(dtype: &DType) -> usize {
        NUM_CHILDREN_NON_NULLABLE + usize::from(dtype.is_nullable())
    }

    /// The writer emits exactly the children the wire contract defines.
    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        vortex_ensure_eq!(nchildren, Self::required_children(dtype));
        Ok(())
    }
}

/// Serialized metadata for the `vortex.list` layout.
///
/// The children are `[elements, offsets]`, followed by a `validity` child exactly when the list
/// dtype is nullable, so every child's position depends only on the dtype:
///
/// - `elements` has the list's element dtype.
/// - `offsets` is a non-nullable integer column of `offsets_ptype` with one more row than the
///   list. Its values index into `elements`.
/// - `validity` is a non-nullable boolean column with one row per list, `true` for valid rows.
///
/// Readers require at least that many children and ignore any that follow. A later writer may
/// append a child under this ID only if reading without it still yields the same values, such as
/// a statistic or an index. A child or metadata field that changes the decoded values needs a new
/// layout ID. Nothing is located relative to the end of the children.
#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    /// The integer [`PType`] of the `offsets` child.
    #[prost(int32, tag = "1")]
    offsets_ptype: i32,
}

impl ListLayoutMetadata {
    /// Metadata for a list layout whose `offsets` child has integer type `offsets_ptype`.
    pub fn new(offsets_ptype: PType) -> Self {
        Self {
            offsets_ptype: offsets_ptype as i32,
        }
    }

    fn get_offsets_ptype(&self) -> VortexResult<PType> {
        let ptype = PType::try_from(self.offsets_ptype).map_err(|_| {
            vortex_err!(
                "ListLayoutMetadata offsets_ptype must be a known PType, got {}",
                self.offsets_ptype
            )
        })?;
        vortex_ensure!(
            ptype.is_int(),
            "ListLayoutMetadata offsets_ptype must be an integer, got {ptype}"
        );
        Ok(ptype)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::List;
    use super::ListLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::LayoutReaderContext;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layout_children;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    fn list_dtype(nullability: Nullability) -> DType {
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            nullability,
        )
    }

    fn u64_offsets_metadata() -> Vec<u8> {
        ListLayoutMetadata::new(PType::U64).encode_to_vec()
    }

    /// A flat layout that is only deserialized, never read.
    fn flat(row_count: u64, dtype: DType) -> LayoutRef {
        FlatLayout::new(row_count, dtype, SegmentId::from(0), ReadContext::new([])).into_layout()
    }

    /// Children for a three-row list of `i32` with five elements and `u64` offsets.
    fn three_row_children(nullability: Nullability) -> Vec<LayoutRef> {
        let mut children = vec![
            flat(5, DType::Primitive(PType::I32, Nullability::NonNullable)),
            flat(4, DType::Primitive(PType::U64, Nullability::NonNullable)),
        ];
        if nullability.is_nullable() {
            children.push(flat(3, DType::Bool(Nullability::NonNullable)));
        }
        children
    }

    fn deserialize(
        session: &VortexSession,
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        children: Vec<LayoutRef>,
    ) -> VortexResult<LayoutRef> {
        let children = layout_children(children);
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session,
            array_read_ctx: &read_ctx,
        };
        LayoutPlugin::deserialize(
            &List,
            LayoutDeserialization {
                serialized_id: LayoutPlugin::id(&List),
                dtype,
                row_count,
                metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &build_ctx,
        )
    }

    fn assert_deserialize_err(
        dtype: &DType,
        row_count: u64,
        metadata: &[u8],
        children: Vec<LayoutRef>,
        message: &str,
    ) {
        let result = deserialize(&SESSION, dtype, row_count, metadata, children);
        assert!(
            result
                .as_ref()
                .is_err_and(|err| err.to_string().contains(message)),
            "expected an error containing {message:?}, got {result:?}"
        );
    }

    fn list_array(nullability: Nullability) -> VortexResult<ArrayRef> {
        let validity = match nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => {
                Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
            }
        };
        let mut ctx = SESSION.create_execution_ctx();
        Ok(ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            validity,
            &mut ctx,
        )?
        .into_array())
    }

    async fn write(
        session: &VortexSession,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let segment_source: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = ListLayoutStrategy::default()
            .write_stream(ArrayContext::empty().into(), segments, stream, eof, session)
            .await?;
        Ok((segment_source, layout))
    }

    async fn read_all(
        session: &VortexSession,
        layout: &LayoutRef,
        segments: Arc<dyn SegmentSource>,
    ) -> VortexResult<ArrayRef> {
        let reader =
            layout.new_reader("".into(), segments, session, &LayoutReaderContext::new())?;
        let row_count = layout.row_count();
        let expr = root().bind(reader.dtype())?;
        reader
            .projection_evaluation(
                &(0..row_count),
                &expr,
                MaskFuture::new_true(usize::try_from(row_count)?),
            )?
            .await
    }

    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    #[tokio::test]
    async fn test_deserialize_round_trip(#[case] nullability: Nullability) -> VortexResult<()> {
        let session = new_session().with_tokio();
        let array = list_array(nullability)?;
        let (segments, written) = write(&session, array.clone()).await?;

        let layout = deserialize(
            &session,
            written.dtype(),
            written.row_count(),
            &written.metadata(),
            written.children()?,
        )?;
        assert_eq!(layout.nchildren(), written.nchildren());

        let read = read_all(&session, &layout, segments).await?;
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(read, array, &mut ctx);
        Ok(())
    }

    /// A later writer may append children that readers can ignore, such as an index.
    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    #[tokio::test]
    async fn test_deserialize_ignores_trailing_children(
        #[case] nullability: Nullability,
    ) -> VortexResult<()> {
        let session = new_session().with_tokio();
        let array = list_array(nullability)?;
        let (segments, written) = write(&session, array.clone()).await?;

        let mut children = written.children()?;
        let required_children = children.len();
        children.push(flat(
            1,
            DType::Primitive(PType::U64, Nullability::NonNullable),
        ));
        let layout = deserialize(
            &session,
            written.dtype(),
            written.row_count(),
            &written.metadata(),
            children,
        )?;
        assert_eq!(layout.nchildren(), required_children + 1);
        assert_eq!(layout.children()?.len(), required_children);

        let read = read_all(&session, &layout, segments).await?;
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(read, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_deserialize_rejects_unknown_offsets_ptype() -> VortexResult<()> {
        let metadata = ListLayoutMetadata { offsets_ptype: 11 }.encode_to_vec();
        assert_deserialize_err(
            &list_dtype(Nullability::NonNullable),
            3,
            &metadata,
            three_row_children(Nullability::NonNullable),
            "ListLayoutMetadata offsets_ptype must be a known PType, got 11",
        );

        assert_eq!(
            ListLayoutMetadata::new(PType::U32).get_offsets_ptype()?,
            PType::U32
        );
        Ok(())
    }

    #[test]
    fn test_deserialize_rejects_non_integer_offsets_ptype() {
        let metadata = ListLayoutMetadata::new(PType::F32).encode_to_vec();
        let mut children = three_row_children(Nullability::NonNullable);
        children[1] = flat(4, DType::Primitive(PType::F32, Nullability::NonNullable));
        assert_deserialize_err(
            &list_dtype(Nullability::NonNullable),
            3,
            &metadata,
            children,
            "ListLayoutMetadata offsets_ptype must be an integer, got f32",
        );
    }

    #[test]
    fn test_deserialize_rejects_non_list_dtype() {
        assert_deserialize_err(
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            3,
            &u64_offsets_metadata(),
            three_row_children(Nullability::NonNullable),
            "ListLayout requires a List dtype",
        );
    }

    #[rstest]
    #[case::non_nullable_without_offsets(Nullability::NonNullable, 1)]
    #[case::nullable_without_validity(Nullability::Nullable, 2)]
    fn test_deserialize_rejects_missing_children(
        #[case] nullability: Nullability,
        #[case] nchildren: usize,
    ) {
        let mut children = three_row_children(nullability);
        children.truncate(nchildren);
        assert_deserialize_err(
            &list_dtype(nullability),
            3,
            &u64_offsets_metadata(),
            children,
            "expects at least",
        );
    }

    #[rstest]
    #[case::elements(0, DType::Primitive(PType::I64, Nullability::NonNullable))]
    #[case::offsets_other_than_metadata(1, DType::Primitive(PType::U32, Nullability::NonNullable))]
    #[case::nullable_validity(2, DType::Bool(Nullability::Nullable))]
    fn test_deserialize_rejects_wrong_child_dtype(#[case] idx: usize, #[case] dtype: DType) {
        let mut children = three_row_children(Nullability::Nullable);
        let row_count = children[idx].row_count();
        children[idx] = flat(row_count, dtype);
        assert_deserialize_err(
            &list_dtype(Nullability::Nullable),
            3,
            &u64_offsets_metadata(),
            children,
            "Child dtype mismatch",
        );
    }

    /// The children hold four offsets, which only fit a three-row list.
    #[rstest]
    #[case::too_few_offsets(4, "List offsets must have one more row than the list")]
    #[case::too_many_offsets(2, "List offsets must have one more row than the list")]
    #[case::overflow(u64::MAX, "overflows its offsets row count")]
    fn test_deserialize_rejects_offsets_row_count(#[case] row_count: u64, #[case] message: &str) {
        assert_deserialize_err(
            &list_dtype(Nullability::NonNullable),
            row_count,
            &u64_offsets_metadata(),
            three_row_children(Nullability::NonNullable),
            message,
        );
    }

    #[test]
    fn test_deserialize_rejects_validity_row_count() {
        let mut children = three_row_children(Nullability::Nullable);
        children[2] = flat(2, DType::Bool(Nullability::NonNullable));
        assert_deserialize_err(
            &list_dtype(Nullability::Nullable),
            3,
            &u64_offsets_metadata(),
            children,
            "List validity row count does not match parent",
        );
    }
}
