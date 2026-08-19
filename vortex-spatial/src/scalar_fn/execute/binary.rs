// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary pairwise dispatch, plus an adapter for row-oriented `geo_types` kernels.

use geo::BoundingRect;
use geo_types::Geometry as GeoGeometry;
use geo_types::Rect;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Execution;
use super::Operand;
use super::geo_types::GeoTypesOutput;
use super::geo_types::eval_column;
use super::geo_types::eval_column_pair;
use crate::extension::decode_geometry_scalar;

/// Dispatch a binary strict geometry kernel over constants and columns.
///
/// A null constant or an empty combined validity mask short-circuits to an all-null constant
/// output. Otherwise, `kernel` receives both operand shapes and the mask of rows where both are
/// valid. Two columns are always paired by row index. The kernel remains responsible for physical
/// input interpretation and Vortex output construction.
pub(crate) fn dispatch_binary<K>(
    left: &ArrayRef,
    right: &ArrayRef,
    output_dtype: DType,
    kernel: K,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    K: FnOnce(Execution<2>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
{
    let len = left.len();
    for operand in [left, right] {
        if operand
            .as_opt::<Constant>()
            .is_some_and(|constant| constant.scalar().is_null())
        {
            return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
        }
    }

    let (left, right, valid) = match (left.as_opt::<Constant>(), right.as_opt::<Constant>()) {
        (Some(left), Some(right)) => (
            Operand::Constant(left.scalar().clone()),
            Operand::Constant(right.scalar().clone()),
            Mask::new_true(len),
        ),
        (Some(left), None) => (
            Operand::Constant(left.scalar().clone()),
            Operand::Column(right.clone()),
            right.validity()?.execute_mask(len, ctx)?,
        ),
        (None, Some(right)) => (
            Operand::Column(left.clone()),
            Operand::Constant(right.scalar().clone()),
            left.validity()?.execute_mask(len, ctx)?,
        ),
        (None, None) => {
            let left_valid = left.validity()?.execute_mask(len, ctx)?;
            let right_valid = right.validity()?.execute_mask(len, ctx)?;
            (
                Operand::Column(left.clone()),
                Operand::Column(right.clone()),
                &left_valid & &right_valid,
            )
        }
    };

    if len != 0 && valid.all_false() {
        return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
    }
    kernel(
        Execution {
            operands: [left, right],
            valid,
            len,
            nullability: output_dtype.nullability(),
        },
        ctx,
    )
}

/// A bounding-rectangle pre-check for [`execute_binary_geo_types`]'s one-constant paths.
///
/// Called per row with rectangles in operand order, it returns `Some(result)` when they prove the
/// result and `None` when the exact kernel must run.
pub(crate) type BboxPrecheck<T> = fn(&Rect<f64>, &Rect<f64>) -> Option<T>;

/// Run a binary row-oriented kernel whose inputs are decoded to `geo_types::Geometry`.
///
/// The `geo_types` name describes the values passed to `compute`, not the output. `T` is converted
/// into a Vortex array before this function returns. Nulls propagate from either operand. With
/// exactly one constant operand, `bbox_precheck` may prove a result from the fixed constant
/// bounding rectangle and the current row's rectangle before the exact kernel runs.
pub(crate) fn execute_binary_geo_types<T, F>(
    left: &ArrayRef,
    right: &ArrayRef,
    compute: F,
    bbox_precheck: Option<BboxPrecheck<T>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoTypesOutput,
    F: Fn(&GeoGeometry<f64>, &GeoGeometry<f64>) -> T + Copy,
{
    let nullability = Nullability::from(left.dtype().is_nullable() || right.dtype().is_nullable());
    dispatch_binary(
        left,
        right,
        T::dtype(nullability),
        |execution, ctx| match execution.operands {
            [Operand::Constant(left), Operand::Constant(right)] => {
                let left = decode_geometry_scalar(&left, ctx)?;
                let right = decode_geometry_scalar(&right, ctx)?;
                Ok(ConstantArray::new(
                    compute(&left, &right).into_scalar(execution.nullability),
                    execution.len,
                )
                .into_array())
            }
            [Operand::Constant(left), Operand::Column(right)] => {
                let left = decode_geometry_scalar(&left, ctx)?;
                let prescreen = bbox_precheck.zip(left.bounding_rect());
                eval_column(
                    &right,
                    &execution.valid,
                    |right| {
                        prescreen
                            .and_then(|(precheck, fixed)| precheck(&fixed, &right.bounding_rect()?))
                            .unwrap_or_else(|| compute(&left, right))
                    },
                    execution.nullability,
                    ctx,
                )
            }
            [Operand::Column(left), Operand::Constant(right)] => {
                let right = decode_geometry_scalar(&right, ctx)?;
                let prescreen = bbox_precheck.zip(right.bounding_rect());
                eval_column(
                    &left,
                    &execution.valid,
                    |left| {
                        prescreen
                            .and_then(|(precheck, fixed)| precheck(&left.bounding_rect()?, &fixed))
                            .unwrap_or_else(|| compute(left, &right))
                    },
                    execution.nullability,
                    ctx,
                )
            }
            [Operand::Column(left), Operand::Column(right)] => eval_column_pair(
                &left,
                &right,
                &execution.valid,
                compute,
                execution.nullability,
                ctx,
            ),
        },
        ctx,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use geo::Contains;
    use geo::Intersects;
    use geo_types::Geometry;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::BboxPrecheck;
    use super::execute_binary_geo_types;
    use crate::test_harness::linestring_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    const DISJOINT_PRECHECK: BboxPrecheck<bool> =
        |left, right| (!left.intersects(right)).then_some(false);

    fn triangle_constant(len: usize, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ring = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (0.0, 0.0)];
        let scalar = polygon_column(vec![vec![ring]])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    fn counting_intersects(
        counter: &Cell<usize>,
    ) -> impl Fn(&Geometry<f64>, &Geometry<f64>) -> bool + Copy {
        move |left, right| {
            counter.set(counter.get() + 1);
            left.intersects(right)
        }
    }

    #[test]
    fn bbox_precheck_skips_exact_test() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let triangle = triangle_constant(3, &mut ctx)?;
        let probes = point_column(vec![50.0, 8.0, 2.0], vec![50.0, 8.0, 2.0])?;
        let exact_runs = Cell::new(0);

        let result = execute_binary_geo_types(
            &triangle,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_PRECHECK),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]), &mut ctx);
        assert_eq!(exact_runs.get(), 2);
        Ok(())
    }

    #[test]
    fn bbox_precheck_leaves_nulls_alone() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let triangle = triangle_constant(3, &mut ctx)?;
        let probes = nullable_point_column(vec![Some((50.0, 50.0)), None, Some((2.0, 2.0))])?;
        let exact_runs = Cell::new(0);

        let result = execute_binary_geo_types(
            &triangle,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_PRECHECK),
            &mut ctx,
        )?;
        let expected = BoolArray::new(
            BitBuffer::from_iter([false, false, true]),
            Validity::from_iter([true, false, true]),
        )
        .into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
        assert_eq!(exact_runs.get(), 1);
        Ok(())
    }

    #[test]
    fn bbox_precheck_sees_rects_in_operand_order() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let probes = point_column(vec![2.0, 50.0], vec![2.0, 50.0])?;
        let triangle = triangle_constant(2, &mut ctx)?;
        let exact_runs = Cell::new(0);
        let counted = |left: &Geometry<f64>, right: &Geometry<f64>| {
            exact_runs.set(exact_runs.get() + 1);
            left.contains(right)
        };

        let result = execute_binary_geo_types(
            &probes,
            &triangle,
            counted,
            Some(|left, right| (!left.contains(right)).then_some(false)),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false]), &mut ctx);
        assert_eq!(exact_runs.get(), 0);
        Ok(())
    }

    #[test]
    fn empty_constant_falls_through_to_exact() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let scalar = linestring_column(vec![vec![]])?.execute_scalar(0, &mut ctx)?;
        let empty = ConstantArray::new(scalar, 2).into_array();
        let probes = point_column(vec![2.0, 50.0], vec![2.0, 50.0])?;
        let exact_runs = Cell::new(0);

        let result = execute_binary_geo_types(
            &empty,
            &probes,
            counting_intersects(&exact_runs),
            Some(DISJOINT_PRECHECK),
            &mut ctx,
        )?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false]), &mut ctx);
        assert_eq!(exact_runs.get(), 2);
        Ok(())
    }

    #[test]
    fn bbox_precheck_matches_exact_results() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let triangle = triangle_constant(6, &mut ctx)?;
        let probes = nullable_point_column(vec![
            Some((50.0, 50.0)),
            Some((8.0, 8.0)),
            Some((2.0, 2.0)),
            None,
            Some((0.0, 0.0)),
            Some((10.0, 0.0)),
        ])?;
        let exact = |left: &Geometry<f64>, right: &Geometry<f64>| left.intersects(right);

        let with_precheck =
            execute_binary_geo_types(&triangle, &probes, exact, Some(DISJOINT_PRECHECK), &mut ctx)?;
        let exact_only = execute_binary_geo_types(&triangle, &probes, exact, None, &mut ctx)?;

        assert_arrays_eq!(with_precheck, exact_only, &mut ctx);
        Ok(())
    }
}
