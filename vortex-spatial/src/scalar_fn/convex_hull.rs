// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_ConvexHull`: the planar convex hull of each native `MultiPoint`.

use geo::ConvexHull;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
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
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::MultiPoint;
use crate::extension::Polygon;
use crate::extension::build_polygon_array;
use crate::extension::coordinate::Dimension;
use crate::extension::geometries;
use crate::extension::polygon_storage_dtype;
use crate::extension::single_geometry;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_unary;

/// Resolve the strict native `MultiPoint -> Polygon` overload.
fn convex_hull_dtype(dtypes: &[DType]) -> VortexResult<ExtDTypeRef> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: convex_hull requires exactly one MultiPoint operand, got {}",
        dtypes.len()
    );
    let Some(input) = dtypes[0].as_extension_opt() else {
        vortex_bail!(
            "spatial: convex_hull operand {} is not a native MultiPoint",
            dtypes[0]
        );
    };
    vortex_ensure!(
        input.is::<MultiPoint>(),
        "spatial: convex_hull operand {} is not a native MultiPoint",
        dtypes[0]
    );

    Ok(ExtDType::<Polygon>::try_new(
        input.metadata::<MultiPoint>().clone(),
        polygon_storage_dtype(Dimension::Xy, dtypes[0].nullability()),
    )?
    .erased())
}

/// Compute hulls for the valid rows and scatter them into a full-length native polygon array.
fn convex_hull_array(
    array: ArrayRef,
    valid: &Mask,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let decoded = geometries(&array.filter(valid.clone())?, ctx)?;
    let hulls = decoded.iter().map(ConvexHull::convex_hull);
    let polygons = match valid.indices() {
        AllOr::All => hulls.map(Some).collect(),
        AllOr::None => vec![None; array.len()],
        AllOr::Some(rows) => {
            let mut polygons = vec![None; array.len()];
            for (&row, hull) in rows.iter().zip(hulls) {
                polygons[row] = Some(hull);
            }
            polygons
        }
    };
    build_polygon_array(
        &polygons,
        output_dtype.metadata::<Polygon>().clone(),
        output_dtype.nullability(),
        &ctx.session().arrow(),
    )
}

/// Execute convex hull after shared unary shape and null dispatch.
fn execute_convex_hull(
    execution: Execution<1, Validity>,
    output_dtype: &ExtDTypeRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(scalar)] => {
            let hull = single_geometry(&scalar, ctx)?.convex_hull();
            let output = build_polygon_array(
                &[Some(hull)],
                output_dtype.metadata::<Polygon>().clone(),
                output_dtype.nullability(),
                &ctx.session().arrow(),
            )?;
            Ok(ConstantArray::new(output.execute_scalar(0, ctx)?, execution.len).into_array())
        }
        [Operand::Column(array)] => {
            let valid = execution.valid.execute_mask(execution.len, ctx)?;
            convex_hull_array(array, &valid, output_dtype, ctx)
        }
    }
}

/// Compute the two-dimensional convex hull of each native `MultiPoint` as a native `Polygon`.
/// Empty, single-point, and collinear inputs remain typed polygons with degenerate exterior rings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialConvexHull;

impl SpatialConvexHull {
    /// A lazy `ScalarFnArray` computing a polygon hull for each native `MultiPoint` row.
    pub fn try_new_array(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialConvexHull, EmptyOptions).erased(),
            vec![array],
        )
    }
}

impl ScalarFnVTable for SpatialConvexHull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.convex_hull");
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
            0 => ChildName::from("multipoint"),
            _ => unreachable!("convex_hull has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Extension(convex_hull_dtype(dtypes)?))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let output_dtype = convex_hull_dtype(std::slice::from_ref(input.dtype()))?;
        dispatch_unary(
            &input,
            DType::Extension(output_dtype.clone()),
            |execution, ctx| execute_convex_hull(execution, &output_dtype, ctx),
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
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::SpatialConvexHull;
    use crate::scalar_fn::area::SpatialArea;
    use crate::scalar_fn::collect::SpatialCollect;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    #[test]
    fn computes_polygon_hulls() -> VortexResult<()> {
        let input = multipoint_column(vec![vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0),
        ]])?;
        let expected = polygon_column(vec![vec![vec![
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
            (2.0, 0.0),
        ]]])?;
        let result = SpatialConvexHull::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::empty(vec![], vec![])]
    #[case::one_point(
        vec![(1.0, 2.0)],
        vec![vec![(1.0, 2.0), (1.0, 2.0)]]
    )]
    #[case::collinear(
        vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        vec![vec![(0.0, 0.0), (2.0, 2.0), (0.0, 0.0)]]
    )]
    fn degenerate_hulls_remain_polygons(
        #[case] points: Vec<(f64, f64)>,
        #[case] expected_rings: Vec<Vec<(f64, f64)>>,
    ) -> VortexResult<()> {
        let input = multipoint_column(vec![points])?;
        let expected = polygon_column(vec![expected_rings])?;
        let result = SpatialConvexHull::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn propagates_nulls() -> VortexResult<()> {
        let input = MaskedArray::try_new(
            multipoint_column(vec![
                vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                vec![(2.0, 2.0)],
            ])?,
            Validity::from_iter([true, false]),
        )?
        .into_array();
        let expected = MaskedArray::try_new(
            polygon_column(vec![
                vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]],
                vec![vec![(2.0, 2.0), (2.0, 2.0)]],
            ])?,
            Validity::from_iter([true, false]),
        )?
        .into_array();
        let result = SpatialConvexHull::try_new_array(input)?.into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn constant_remains_constant() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let scalar = multipoint_column(vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])?
            .execute_scalar(0, &mut ctx)?;
        let input = ConstantArray::new(scalar, 3).into_array();

        let result = SpatialConvexHull::try_new_array(input)?.into_array();
        let Columnar::Constant(constant) = result.clone().execute::<Columnar>(&mut ctx)? else {
            return Err(vortex_err!(
                "convex_hull of a constant should remain constant"
            ));
        };
        assert_eq!(constant.len(), 3);
        let hull = vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]];
        let expected = polygon_column(vec![hull.clone(), hull.clone(), hull])?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn collect_hull_area_pipeline() -> VortexResult<()> {
        let points = point_column(
            vec![0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0],
        )?;
        let point_lists = ListArray::try_new(
            points,
            PrimitiveArray::from_iter([0_u32, 5, 8]).into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let collected = SpatialCollect::try_new_array(point_lists)?.into_array();
        let hulls = SpatialConvexHull::try_new_array(collected)?.into_array();
        let areas = SpatialArea::try_new_array(hulls)?.into_array();
        let expected = PrimitiveArray::from_iter([4.0_f64, 0.0]).into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::none(0)]
    #[case::two(2)]
    fn rejects_wrong_arity(#[case] arity: usize) -> VortexResult<()> {
        let dtype = multipoint_column(vec![vec![]])?.dtype().clone();
        assert!(
            SpatialConvexHull
                .return_dtype(&EmptyOptions, &vec![dtype; arity])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_multipoint_input() -> VortexResult<()> {
        let input: ArrayRef = point_column(vec![0.0], vec![0.0])?;
        assert!(SpatialConvexHull::try_new_array(input).is_err());
        Ok(())
    }
}
