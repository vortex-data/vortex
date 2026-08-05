// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Collect`: collect homogeneous native geometries into their native multi-geometry type.

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
fn collect_dtype(dtypes: &[DType]) -> VortexResult<DType> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: collect requires exactly one list operand, got {}",
        dtypes.len()
    );
    let DType::List(element_dtype, nullability) = &dtypes[0] else {
        vortex_bail!("spatial: collect operand {} is not a list", dtypes[0]);
    };
    let Some(element) = element_dtype.as_extension_opt() else {
        vortex_bail!(
            "spatial: collect list element {} is not a native Point, LineString, or Polygon",
            element_dtype
        );
    };
    // Multi-geometries cannot contain null components. Null list elements are ignored during
    // execution, so their storage is non-nullable in the result.
    let storage = DType::List(
        Arc::new(element.storage_dtype().as_nonnullable()),
        *nullability,
    );
    let output = if element.is::<Point>() {
        ExtDType::<MultiPoint>::try_new(element.metadata::<Point>().clone(), storage)?.erased()
    } else if element.is::<LineString>() {
        ExtDType::<MultiLineString>::try_new(element.metadata::<LineString>().clone(), storage)?
            .erased()
    } else if element.is::<Polygon>() {
        ExtDType::<MultiPolygon>::try_new(element.metadata::<Polygon>().clone(), storage)?.erased()
    } else {
        vortex_bail!(
            "spatial: collect list element {} is not a native Point, LineString, or Polygon",
            element_dtype
        );
    };
    Ok(DType::Extension(output))
}

/// Count valid elements in an exact list row without per-element mask lookups.
fn valid_count(mask: &Mask, start: usize, end: usize) -> usize {
    match mask.bit_buffer() {
        AllOr::All => end - start,
        AllOr::None => 0,
        AllOr::Some(bits) => bits.count_range(start, end),
    }
}

/// Rewrap a homogeneous geometry list as its corresponding multi-geometry array.
///
/// The all-valid path reuses the geometry payload and list views. If geometry elements are null,
/// DuckDB semantics require ignoring them; that path first makes the views exact, then compacts the
/// payload and rebuilds the row views.
fn collect_list(
    mut list: ListViewArray,
    validity: Validity,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut element_valid = list
        .elements()
        .validity()?
        .execute_mask(list.elements().len(), ctx)?;
    if !element_valid.all_true() {
        list = list.rebuild(ListViewRebuildMode::MakeExact, ctx)?;
        element_valid = list
            .elements()
            .validity()?
            .execute_mask(list.elements().len(), ctx)?;
    }

    let parts = list.into_data_parts();
    let elements = parts.elements.execute::<ExtensionArray>(ctx)?;
    let DType::List(target_element_storage, _) = output_dtype.storage_dtype() else {
        unreachable!("collect output storage is always a list")
    };
    let target_element_storage = target_element_storage.as_ref().clone();

    let compact_elements = !element_valid.all_true();
    let element_storage = if compact_elements {
        elements
            .storage_array()
            .filter(element_valid.clone())?
            .cast(target_element_storage)?
    } else {
        elements.storage_array().cast(target_element_storage)?
    };

    let (offsets, sizes) = if compact_elements {
        let old_offsets = parts
            .offsets
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        let old_sizes = parts
            .sizes
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        let mut offsets = BufferMut::<u64>::with_capacity(old_offsets.len());
        let mut sizes = BufferMut::<u64>::with_capacity(old_sizes.len());
        let mut next_offset = 0_u64;

        for (&old_offset, &old_size) in old_offsets.iter().zip(old_sizes.iter()) {
            let start = usize::try_from(old_offset)
                .map_err(|_| vortex_err!("spatial: collect element offset exceeds usize"))?;
            let size = usize::try_from(old_size)
                .map_err(|_| vortex_err!("spatial: collect element count exceeds usize"))?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| vortex_err!("spatial: collect element range overflows usize"))?;
            vortex_ensure!(
                end <= element_valid.len(),
                "spatial: collect element range {start}..{end} exceeds element length {}",
                element_valid.len()
            );
            let size = u64::try_from(valid_count(&element_valid, start, end))
                .map_err(|_| vortex_err!("spatial: collect valid element count exceeds u64"))?;
            offsets.push(next_offset);
            sizes.push(size);
            next_offset = next_offset
                .checked_add(size)
                .ok_or_else(|| vortex_err!("spatial: collect output offset exceeds u64"))?;
        }
        (offsets.into_array(), sizes.into_array())
    } else {
        (parts.offsets, parts.sizes)
    };

    let storage = ListViewArray::try_new(element_storage, offsets, sizes, validity)?.into_array();
    Ok(ExtensionArray::try_new(output_dtype.clone(), storage)?.into_array())
}

/// Execute the structural collect kernel after shared unary shape and null dispatch.
fn execute_collect(
    execution: Execution<1>,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(scalar)] => {
            let one = ConstantArray::new(scalar, 1)
                .into_array()
                .execute::<ListViewArray>(ctx)?;
            let collected = collect_list(
                one,
                Validity::from_mask(Mask::new_true(1), output_dtype.nullability()),
                output_dtype,
                ctx,
            )?;
            Ok(ConstantArray::new(collected.execute_scalar(0, ctx)?, execution.len).into_array())
        }
        [Operand::Column(array)] => collect_list(
            array.execute::<ListViewArray>(ctx)?,
            Validity::from_mask(execution.valid, output_dtype.nullability()),
            output_dtype,
            ctx,
        ),
    }
}

/// Collect a homogeneous list of native `Point`, `LineString`, or `Polygon` values into the
/// corresponding `MultiPoint`, `MultiLineString`, or `MultiPolygon` value. Null geometry elements
/// are ignored. Mixed geometry lists are rejected by the list element dtype rather than represented
/// as a geometry union.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialCollect;

impl SpatialCollect {
    /// A lazy `ScalarFnArray` collecting each list row into one native multi-geometry value.
    pub fn try_new_array(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialCollect, EmptyOptions).erased(),
            vec![array],
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
            0 => ChildName::from("geometries"),
            _ => unreachable!("collect has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        collect_dtype(dtypes)
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let output_dtype = collect_dtype(std::slice::from_ref(input.dtype()))?;
        let output = output_dtype.as_extension().clone();
        dispatch_unary(
            &input,
            output_dtype,
            |execution, ctx| execute_collect(execution, &output, ctx),
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

    #[test]
    fn collects_points_into_multipoints() -> VortexResult<()> {
        let points = point_column(vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])?;
        let input = list(points, &[0, 2, 3])?;
        let expected = multipoint_column(vec![vec![(0.0, 3.0), (1.0, 4.0)], vec![(2.0, 5.0)]])?;
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
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

        let result = SpatialCollect::try_new_array(input)?
            .into_array()
            .execute::<ExtensionArray>(&mut ctx)?;
        let result_storage = result
            .storage_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)?;

        assert!(ArrayRef::ptr_eq(&point_storage, result_storage.elements()));
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
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
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
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
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

        let result = SpatialCollect::try_new_array(input)?.into_array();
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
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn all_null_geometry_elements_produce_empty_multi_geometry() -> VortexResult<()> {
        let input = list(nullable_point_column(vec![None, None])?, &[0, 2])?;
        let expected = multipoint_column(vec![vec![]])?;
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
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
        let result = SpatialCollect::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_unsupported_inputs() -> VortexResult<()> {
        let point = point_column(vec![0.0], vec![0.0])?;
        assert!(SpatialCollect::try_new_array(point).is_err());

        let multipoints = multipoint_column(vec![vec![(0.0, 0.0)]])?;
        assert!(SpatialCollect::try_new_array(list(multipoints, &[0, 1])?).is_err());

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
