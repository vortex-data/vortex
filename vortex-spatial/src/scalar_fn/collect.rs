// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-row scalar `ST_Collect` over homogeneous native geometry lists.
//!
//! One `List<Point>`, `List<LineString>`, or `List<Polygon>` row in; one `MultiPoint`,
//! `MultiLineString`, or `MultiPolygon` row out. This is not an aggregate: it never combines
//! geometries from different rows, so a query over individual geometry rows must group them
//! first with `ARRAY_AGG` or `list`.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::LineString;
use crate::extension::MultiLineString;
use crate::extension::MultiPoint;
use crate::extension::MultiPolygon;
use crate::extension::Point;
use crate::extension::Polygon;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_unary;

/// Resolve the strict homogeneous `ST_Collect` overload for one list operand.
fn collect_dtype(dtypes: &[DType]) -> VortexResult<ExtDTypeRef> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: collect requires exactly one list operand, got {}",
        dtypes.len()
    );
    let DType::List(element_dtype, nullability) = &dtypes[0] else {
        vortex_bail!("spatial: collect operand {} is not a list", dtypes[0]);
    };
    // Execution ignores null list elements, so the result's element storage is non-nullable.
    let multi_storage = |element: &ExtDTypeRef| {
        DType::List(
            Arc::new(element.storage_dtype().as_nonnullable()),
            *nullability,
        )
    };
    match element_dtype.as_extension_opt() {
        Some(element) if element.is::<Point>() => Ok(ExtDType::<MultiPoint>::try_new(
            element.metadata::<Point>().clone(),
            multi_storage(element),
        )?
        .erased()),
        Some(element) if element.is::<LineString>() => Ok(ExtDType::<MultiLineString>::try_new(
            element.metadata::<LineString>().clone(),
            multi_storage(element),
        )?
        .erased()),
        Some(element) if element.is::<Polygon>() => Ok(ExtDType::<MultiPolygon>::try_new(
            element.metadata::<Polygon>().clone(),
            multi_storage(element),
        )?
        .erased()),
        _ => vortex_bail!(
            "spatial: collect list element {element_dtype} is not a native Point, LineString, \
             or Polygon"
        ),
    }
}

/// Count valid elements in `start..end` with one range popcount, not per-element lookups.
fn count_valid(element_mask: &Mask, start: usize, end: usize) -> usize {
    match element_mask.bit_buffer() {
        AllOr::All => end - start,
        AllOr::None => 0,
        AllOr::Some(bits) => bits.count_range(start, end),
    }
}

/// Re-address the rows once null elements are filtered out of the payload.
///
/// Takes the `row_sizes` of an exact list view and the mask over its elements; returns the new
/// `(offsets, sizes)`. The old offsets are redundant: `MakeExact` leaves the views a gapless
/// in-order cover, so each row starts where the previous one ended. Neither running sum can
/// exceed the element count.
fn compact_row_views(
    row_sizes: &ArrayRef,
    element_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let sizes = row_sizes
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<Buffer<u64>>(ctx)?;
    let mut compact_offsets = BufferMut::<u64>::with_capacity(sizes.len());
    let mut compact_sizes = BufferMut::<u64>::with_capacity(sizes.len());
    let mut start = 0usize;
    let mut offset = 0_u64;

    for &size in sizes.iter() {
        let end = usize::try_from(size)
            .ok()
            .and_then(|row_len| start.checked_add(row_len))
            .filter(|end| *end <= element_mask.len())
            .ok_or_else(|| {
                vortex_err!(
                    "spatial: collect row at element {start} exceeds the {} list elements",
                    element_mask.len()
                )
            })?;
        let valid = u64::try_from(count_valid(element_mask, start, end))
            .map_err(|_| vortex_err!("spatial: collect valid element count exceeds u64"))?;
        compact_offsets.push(offset);
        compact_sizes.push(valid);
        offset += valid;
        start = end;
    }
    Ok((compact_offsets.into_array(), compact_sizes.into_array()))
}

/// Rewrap each geometry-list row as one multi-geometry row.
///
/// All-valid rows reuse the payload and views untouched. Null elements must be ignored (DuckDB
/// semantics), so that path makes the views exact, filters the payload, and re-addresses the rows.
/// Both paths forward the input's zero-copy-to-list flag.
fn collect_list_rows(
    mut lists: ListViewArray,
    validity: Validity,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut element_mask = lists
        .elements()
        .validity()?
        .execute_mask(lists.elements().len(), ctx)?;
    if !element_mask.all_true() {
        // Filtering addresses elements by position, so make the views exact first. `MakeExact`
        // re-gathers the elements, so the mask must be taken again.
        lists = lists.rebuild(ListViewRebuildMode::MakeExact, ctx)?;
        element_mask = lists
            .elements()
            .validity()?
            .execute_mask(lists.elements().len(), ctx)?;
    }

    // Both paths keep the views exact — one forwards them, the other rebuilds them as a running
    // sum in the same element order — so the output is zero-copy to a `ListArray` whenever the
    // input is.
    let zero_copy_to_list = lists.is_zero_copy_to_list();
    let parts = lists.into_data_parts();
    let elements = parts.elements.execute::<ExtensionArray>(ctx)?;

    let (element_storage, offsets, sizes) = if element_mask.all_true() {
        (elements.storage_array().clone(), parts.offsets, parts.sizes)
    } else {
        let (offsets, sizes) = compact_row_views(&parts.sizes, &element_mask, ctx)?;
        (
            elements.storage_array().filter(element_mask)?,
            offsets,
            sizes,
        )
    };

    let output_element_dtype = output_dtype
        .storage_dtype()
        .as_list_element_opt()
        .vortex_expect("collect output storage is always a list")
        .as_ref()
        .clone();
    let storage = ListViewArray::try_new(
        element_storage.cast(output_element_dtype)?,
        offsets,
        sizes,
        validity,
    )?;
    // SAFETY: the views were either forwarded unchanged or rebuilt as a gapless, non-overlapping
    // running sum over the same elements, so the flag still holds. Forwarding it matters:
    // `list_from_list_view` re-gathers the whole payload when it reads `false`.
    let storage = unsafe { storage.with_zero_copy_to_list(zero_copy_to_list) }.into_array();
    Ok(ExtensionArray::try_new(output_dtype.clone(), storage)?.into_array())
}

/// Apply [`collect_list_rows`] to a constant or column, after shared unary null dispatch.
fn execute_collect(
    execution: Execution<1, Validity>,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(scalar)] => {
            let one_list = ConstantArray::new(scalar, 1)
                .into_array()
                .execute::<ListViewArray>(ctx)?;
            let collected = collect_list_rows(
                one_list,
                Validity::from_mask(Mask::new_true(1), execution.nullability),
                output_dtype,
                ctx,
            )?;
            Ok(ConstantArray::new(collected.execute_scalar(0, ctx)?, execution.len).into_array())
        }
        [Operand::Column(geometry_lists)] => {
            let valid = execution.valid.execute_mask(execution.len, ctx)?;
            collect_list_rows(
                geometry_lists.execute::<ListViewArray>(ctx)?,
                Validity::from_mask(valid, execution.nullability),
                output_dtype,
                ctx,
            )
        }
    }
}

/// Scalar `ST_Collect`: one `List<Point>`, `List<LineString>`, or `List<Polygon>` row in, one
/// `MultiPoint`, `MultiLineString`, or `MultiPolygon` row out.
///
/// Not an aggregate — it never combines rows. Null elements are ignored, and mixed geometry lists
/// are rejected by the element dtype rather than widened to a union.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialCollect;

impl SpatialCollect {
    /// Create a lazy scalar array that converts each geometry-list row to one multi-geometry row.
    pub fn try_new(geometry_lists: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialCollect, EmptyOptions).erased(),
            vec![geometry_lists],
        )
    }
}

impl ScalarFnVTable for SpatialCollect {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.collect");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("geometry_list"),
            _ => unreachable!("collect has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Extension(collect_dtype(dtypes)?))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let geometry_lists = args.get(0)?;
        let output_dtype = collect_dtype(std::slice::from_ref(geometry_lists.dtype()))?;
        dispatch_unary(
            &geometry_lists,
            DType::Extension(output_dtype.clone()),
            |execution, ctx| execute_collect(execution, &output_dtype, ctx),
            ctx,
        )
    }

    fn validity(
        &self,
        _: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Columnar;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::listview::ListViewArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::SpatialCollect;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    fn list_with_validity(
        elements: ArrayRef,
        offsets: &[u32],
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        Ok(ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets.iter().copied()).into_array(),
            validity,
        )?
        .into_array())
    }

    fn list(elements: ArrayRef, offsets: &[u32]) -> VortexResult<ArrayRef> {
        list_with_validity(elements, offsets, Validity::NonNullable)
    }

    /// Assert that `ST_Collect` of `input` equals `expected`.
    fn assert_collects(input: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let result = SpatialCollect::try_new(input)?.into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn collects_points_into_multipoints() -> VortexResult<()> {
        let points = point_column(vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])?;
        let input = list(points, &[0, 2, 3])?;
        let expected = multipoint_column(vec![vec![(0.0, 3.0), (1.0, 4.0)], vec![(2.0, 5.0)]])?;

        assert_collects(input, expected)
    }

    /// Out-of-order rows make the view non-exact. `compact_row_views` derives offsets from `sizes`
    /// alone, which only holds after `MakeExact` reorders the elements.
    #[test]
    fn collects_out_of_order_list_views() -> VortexResult<()> {
        let points = nullable_point_column(vec![
            Some((0.0, 4.0)),
            None,
            Some((2.0, 6.0)),
            Some((3.0, 7.0)),
        ])?;
        let views = ListViewArray::try_new(
            points,
            PrimitiveArray::from_iter([2u32, 0]).into_array(),
            PrimitiveArray::from_iter([2u32, 2]).into_array(),
            Validity::NonNullable,
        )?;
        assert!(
            !views.is_zero_copy_to_list(),
            "out-of-order row views are not zero-copy to a list"
        );
        let expected = multipoint_column(vec![vec![(2.0, 6.0), (3.0, 7.0)], vec![(0.0, 4.0)]])?;

        assert_collects(views.into_array(), expected)
    }

    #[test]
    fn all_valid_collect_reuses_geometry_storage() -> VortexResult<()> {
        let points = point_column(vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])?;
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let point_storage = points
            .clone()
            .execute::<ExtensionArray>(&mut ctx)?
            .storage_array()
            .clone();
        let input = list(points, &[0, 2, 3])?;

        let result = SpatialCollect::try_new(input)?
            .into_array()
            .execute::<ExtensionArray>(&mut ctx)?;
        let result_storage = result
            .storage_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)?;

        assert!(ArrayRef::ptr_eq(&point_storage, result_storage.elements()));
        Ok(())
    }

    /// A list view that forgets it is zero-copy to a list makes the next
    /// `list_from_list_view` re-gather the payload that collect just reused.
    #[rstest]
    #[case::reused_elements(false)]
    #[case::compacted_elements(true)]
    fn output_stays_zero_copy_to_list(#[case] null_elements: bool) -> VortexResult<()> {
        let points = if null_elements {
            nullable_point_column(vec![Some((0.0, 3.0)), None, Some((2.0, 5.0))])?
        } else {
            point_column(vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])?
        };
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let input = list(points, &[0, 2, 3])?;
        assert!(
            input
                .clone()
                .execute::<ListViewArray>(&mut ctx)?
                .is_zero_copy_to_list(),
            "a list column reaches collect as an exact list view"
        );

        let storage = SpatialCollect::try_new(input)?
            .into_array()
            .execute::<ExtensionArray>(&mut ctx)?
            .storage_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)?;

        assert!(storage.is_zero_copy_to_list());
        Ok(())
    }

    #[test]
    fn collects_linestrings_into_multilinestrings() -> VortexResult<()> {
        let line_a = vec![(0.0, 0.0), (1.0, 1.0)];
        let line_b = vec![(2.0, 2.0), (3.0, 3.0)];
        let line_c = vec![(4.0, 4.0), (5.0, 5.0)];
        let input = list(
            linestring_column(vec![line_a.clone(), line_b.clone(), line_c.clone()])?,
            &[0, 2, 3],
        )?;
        let expected = multilinestring_column(vec![vec![line_a, line_b], vec![line_c]])?;

        assert_collects(input, expected)
    }

    #[test]
    fn collects_polygons_into_multipolygons() -> VortexResult<()> {
        let polygon_a = vec![vec![(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]];
        let polygon_b = vec![vec![(3.0, 0.0), (5.0, 0.0), (3.0, 2.0), (3.0, 0.0)]];
        let polygon_c = vec![vec![(6.0, 0.0), (8.0, 0.0), (6.0, 2.0), (6.0, 0.0)]];
        let input = list(
            polygon_column(vec![
                polygon_a.clone(),
                polygon_b.clone(),
                polygon_c.clone(),
            ])?,
            &[0, 2, 3],
        )?;
        let expected = multipolygon_column(vec![vec![polygon_a, polygon_b], vec![polygon_c]])?;

        assert_collects(input, expected)
    }

    #[test]
    fn constant_list_remains_constant() -> VortexResult<()> {
        let input = list(
            nullable_point_column(vec![Some((0.0, 2.0)), None, Some((1.0, 3.0))])?,
            &[0, 3],
        )?;
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let scalar = input.execute_scalar(0, &mut ctx)?;
        let input = ConstantArray::new(scalar, 3).into_array();

        let result = SpatialCollect::try_new(input)?.into_array();
        let Columnar::Constant(constant) = result.clone().execute::<Columnar>(&mut ctx)? else {
            return Err(vortex_err!(
                "collect of a constant list should remain constant"
            ));
        };
        assert_eq!(constant.len(), 3);
        let expected = multipoint_column(vec![
            vec![(0.0, 2.0), (1.0, 3.0)],
            vec![(0.0, 2.0), (1.0, 3.0)],
            vec![(0.0, 2.0), (1.0, 3.0)],
        ])?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn ignores_null_geometry_elements() -> VortexResult<()> {
        let points = nullable_point_column(vec![Some((0.0, 2.0)), None, Some((1.0, 3.0)), None])?;
        let input = list(points, &[0, 2, 4])?;
        let expected = multipoint_column(vec![vec![(0.0, 2.0)], vec![(1.0, 3.0)]])?;

        assert_collects(input, expected)
    }

    #[test]
    fn all_null_geometry_elements_produce_empty_multi_geometry() -> VortexResult<()> {
        let input = list(nullable_point_column(vec![None, None])?, &[0, 2])?;
        let expected = multipoint_column(vec![vec![]])?;

        assert_collects(input, expected)
    }

    #[test]
    fn propagates_null_list_rows() -> VortexResult<()> {
        let input = list_with_validity(
            point_column(vec![0.0, 1.0], vec![2.0, 3.0])?,
            &[0, 1, 2],
            Validity::from_iter([true, false]),
        )?;
        let expected = MaskedArray::try_new(
            multipoint_column(vec![vec![(0.0, 2.0)], vec![(1.0, 3.0)]])?,
            Validity::from_iter([true, false]),
        )?
        .into_array();

        assert_collects(input, expected)
    }

    #[test]
    fn rejects_unsupported_inputs() -> VortexResult<()> {
        let point = point_column(vec![0.0], vec![0.0])?;
        assert!(SpatialCollect::try_new(point).is_err());

        let multipoints = multipoint_column(vec![vec![(0.0, 0.0)]])?;
        assert!(SpatialCollect::try_new(list(multipoints, &[0, 1])?).is_err());

        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(
            SpatialCollect
                .return_dtype(
                    &EmptyOptions,
                    &[DType::List(primitive.into(), Nullability::NonNullable)]
                )
                .is_err()
        );
        Ok(())
    }
}
