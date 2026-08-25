// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Distance`: planar (Euclidean) distance between two native geometries.

use geo::Distance;
use geo::Euclidean;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::scalar_fn::row::GeometryRow;

/// Planar (Euclidean) `ST_Distance` (no geodesic correction) between two native geometry
/// operands, each a column or a constant literal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialDistance;

impl SpatialDistance {
    /// A lazy `ScalarFnArray` computing the per-row distance between operands `a` and `b`; either may
    /// be constant. The output length is taken from `a`.
    pub fn try_new(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialDistance, EmptyOptions).erased(),
            vec![a, b],
        )
    }
}

impl RowFn for SpatialDistance {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["a", "b"];
    // The per-row distance cannot fail, but decoding a geometry operand can.
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.distance");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        // Distance is a value, not a verdict: no bounding-rect test can decide it.
        visitor.visit::<(GeometryRow, GeometryRow), f64>(|(a, b)| Euclidean.distance(a, b))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::SpatialDistance;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    /// A constant `Point` column of length `len`, every row at `(x, y)`.
    fn point_constant(
        x: f64,
        y: f64,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let single = point_column(vec![x], vec![y])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(single, len).into_array())
    }

    /// Execute a `SpatialDistance` array and read back its per-row `f64` distances.
    fn distances(distance: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<f64>> {
        Ok(distance
            .execute::<Canonical>(ctx)?
            .into_primitive()
            .as_slice::<f64>()
            .to_vec())
    }

    /// `SpatialDistance` returns the per-row distance between a point column and a constant query point
    /// (3–4–5 triangles), computed via the external `geo` crate.
    #[test]
    fn distance_over_points() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![0.0, 3.0, 0.0, 3.0], vec![0.0, 0.0, 4.0, 4.0])?;
        let b = point_constant(0.0, 0.0, 4, &mut ctx)?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![0.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    /// Column-to-column distance pairs corresponding rows of the two columns.
    #[test]
    fn distance_between_columns() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![0.0, 1.0], vec![0.0, 1.0])?;
        let b = point_column(vec![3.0, 1.0], vec![4.0, 1.0])?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![5.0, 0.0]);
        Ok(())
    }

    /// Distance passes no bounding-rect rejection: a point far outside a constant polygon's
    /// bounding rect still gets its true distance, alongside an inside point at distance zero.
    #[test]
    fn distance_to_constant_polygon_is_exact() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let ring = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)];
        let single = polygon_column(vec![vec![ring]])?.execute_scalar(0, &mut ctx)?;
        let square = ConstantArray::new(single, 2).into_array();
        let points = point_column(vec![7.0, 2.0], vec![2.0, 2.0])?;
        let distance = SpatialDistance::try_new(points, square)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![3.0, 0.0]);
        Ok(())
    }

    /// The constant query point may be either operand; distance is symmetric.
    #[test]
    fn distance_with_constant_first_operand() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_constant(0.0, 0.0, 4, &mut ctx)?;
        let b = point_column(vec![0.0, 3.0, 0.0, 3.0], vec![0.0, 0.0, 4.0, 4.0])?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![0.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    /// Two constant operands: every row has the same distance.
    #[test]
    fn distance_between_two_constants() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_constant(0.0, 0.0, 3, &mut ctx)?;
        let b = point_constant(3.0, 4.0, 3, &mut ctx)?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![5.0, 5.0, 5.0]);
        Ok(())
    }

    /// Output nullability mirrors the operands: nullable if any operand is nullable, otherwise
    /// non-nullable.
    #[test]
    fn output_nullability_mirrors_operands() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let non_nullable =
            SpatialDistance.return_dtype(&EmptyOptions, &[dtype.clone(), dtype.clone()])?;
        assert!(!non_nullable.is_nullable());
        let nullable =
            SpatialDistance.return_dtype(&EmptyOptions, &[dtype.as_nullable(), dtype])?;
        assert!(nullable.is_nullable());
        Ok(())
    }

    /// A null row in a geometry operand yields a null result; valid rows are unaffected.
    #[test]
    fn distance_propagates_null_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![Some((0.0, 0.0)), None, Some((3.0, 4.0))])?;
        let b = point_constant(0.0, 0.0, 3, &mut ctx)?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        let expected = PrimitiveArray::new(
            vec![0.0f64, 0.0, 5.0],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// A nullable column of a list-storage geometry type (`MultiPolygon`): nulls propagate and
    /// valid rows get exact distances, with the null row's storage never decoded.
    #[test]
    fn distance_propagates_nulls_for_multipolygon() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let near = vec![vec![vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]]];
        let far = vec![vec![vec![
            (10.0, 6.0),
            (14.0, 6.0),
            (14.0, 10.0),
            (10.0, 10.0),
            (10.0, 6.0),
        ]]];
        let a = nullable_multipolygon_column(vec![Some(near), None, Some(far)])?;
        let b = point_constant(7.0, 2.0, 3, &mut ctx)?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        let expected = PrimitiveArray::new(
            vec![3.0f64, 0.0, 5.0],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// Both operands nullable: a row is null if either operand is null there.
    #[test]
    fn distance_propagates_column_pair_nulls() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![Some((0.0, 0.0)), None, Some((0.0, 0.0))])?;
        let b = nullable_point_column(vec![Some((3.0, 4.0)), Some((1.0, 1.0)), None])?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        let expected = PrimitiveArray::new(
            vec![5.0f64, 0.0, 0.0],
            Validity::from_iter([true, false, false]),
        )
        .into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// A constant-null operand produces an all-null output.
    #[test]
    fn distance_constant_null_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let null_const = ConstantArray::new(Scalar::null(point_dtype), 3).into_array();
        let b = point_column(vec![0.0, 3.0, 0.0], vec![0.0, 0.0, 4.0])?;
        let distance = SpatialDistance::try_new(null_const, b)?.into_array();

        let expected = PrimitiveArray::new(vec![0.0f64; 3], Validity::AllInvalid).into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// An entirely-null geometry column yields an all-null output.
    #[test]
    fn distance_all_null_column_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![None, None])?;
        let b = point_constant(0.0, 0.0, 2, &mut ctx)?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        let expected = PrimitiveArray::new(vec![0.0f64; 2], Validity::AllInvalid).into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// Two nullable columns whose nulls never line up: the combined mask is empty, so the output
    /// is all null.
    #[test]
    fn distance_column_pair_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![Some((0.0, 0.0)), None])?;
        let b = nullable_point_column(vec![None, Some((1.0, 1.0))])?;
        let distance = SpatialDistance::try_new(a, b)?.into_array();

        let expected = PrimitiveArray::new(vec![0.0f64; 2], Validity::AllInvalid).into_array();
        assert_arrays_eq!(distance, expected, &mut ctx);
        Ok(())
    }

    /// A zero-length non-nullable execution keeps the non-nullable result dtype: an empty
    /// `AllTrue` mask has `true_count() == 0`, and must not be mistaken for all-null and widened
    /// to nullable (which would trip the result-dtype assertion against `return_dtype`).
    #[test]
    fn distance_empty_non_nullable_keeps_dtype() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![], vec![])?;
        let b = point_column(vec![], vec![])?;
        let result = SpatialDistance::try_new(a, b)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert!(!result.dtype().is_nullable());
        assert_eq!(result.len(), 0);
        Ok(())
    }

    /// A non-geometry operand dtype is rejected up front, before execution.
    #[test]
    fn non_geometry_operand_is_rejected() -> VortexResult<()> {
        let spatial_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = SpatialDistance.return_dtype(&EmptyOptions, &[spatial_dtype, numeric]);
        assert!(result.is_err());
        Ok(())
    }
}
