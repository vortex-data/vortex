// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Contains`: OGC containment test between two native geometries.

use geo::Contains;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::is_native_geometry;
use crate::scalar_fn::execute::execute_binary_geo_types;

/// Validate the two native geometry operands accepted by `ST_Contains`.
fn validate_contains_operands(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 2,
        "spatial: contains requires exactly two geometry operands, got {}",
        dtypes.len()
    );
    for dtype in dtypes {
        vortex_ensure!(
            is_native_geometry(dtype),
            "spatial: contains operand {dtype} is not a native geometry type"
        );
    }
    Ok(())
}

/// OGC `ST_Contains` between two native geometry operands, each a column or a constant
/// literal: true where operand `b` lies completely inside operand `a` (boundary contact alone
/// does not count). Containment is not symmetric; the operand order is significant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialContains;

impl SpatialContains {
    /// A lazy `ScalarFnArray` computing per-row whether operand `a` contains operand `b`;
    /// either may be constant. The output length is taken from `a`.
    pub fn try_new(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialContains, EmptyOptions).erased(),
            vec![a, b],
        )
    }
}

impl ScalarFnVTable for SpatialContains {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.contains");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("a"),
            1 => ChildName::from("b"),
            _ => unreachable!("contains has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_contains_operands(dtypes)?;
        let nullability = Nullability::from(dtypes.iter().any(DType::is_nullable));
        Ok(DType::Bool(nullability))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let a = args.get(0)?;
        let b = args.get(1)?;
        // Containment is not symmetric: `a` is always the container and `b` the contained. A
        // container's rect must cover the contained's rect (`Rect::contains` is the closed
        // test), so a contained rect poking outside proves the row false.
        execute_binary_geo_types(
            &a,
            &b,
            |a, b| a.contains(b),
            Some(|ra, rb| (!ra.contains(rb)).then_some(false)),
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
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::Point;
    use geo_types::Polygon;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use wkb::writer::WriteOptions;

    use super::SpatialContains;
    use crate::test_harness::linestring_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;

    /// A rectangle polygon with corners `(x0, y0)` and `(x1, y1)`, no holes.
    fn rect_polygon(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(
            LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
            vec![],
        )
    }

    /// A constant column of length `len`, every row the native form of `geometry`.
    fn geometry_constant(geometry: &Geometry, len: usize) -> VortexResult<ArrayRef> {
        let mut buf = Vec::new();
        wkb::writer::write_geometry(&mut buf, geometry, &WriteOptions::default())
            .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;
        let session = vortex_array::array_session();
        let scalar = crate::extension::native_geometry_scalar_from_wkb(&buf, &session.arrow())?
            .ok_or_else(|| vortex_err!("unsupported geometry type"))?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    /// Materialize `array` so it is no longer a `Constant`, forcing the non-constant kernel
    /// paths.
    fn materialize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.execute::<Canonical>(ctx)?.into_array())
    }

    /// Execute `SpatialContains(a, b)` and assert the per-row verdicts equal `expected`.
    fn assert_contains(
        a: ArrayRef,
        b: ArrayRef,
        expected: impl IntoIterator<Item = bool>,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let contains = SpatialContains::try_new(a, b)?.into_array();
        assert_arrays_eq!(contains, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    // The tests cover each `execute` dispatch arm in match order, then the edge cases.

    /// Constant vs constant: a polygon contains a nested polygon but not a partially
    /// overlapping or disjoint one; every output row carries the same verdict.
    #[rstest]
    #[case::nested(rect_polygon(1.0, 1.0, 3.0, 3.0), true)]
    #[case::overlapping(rect_polygon(2.0, 2.0, 6.0, 6.0), false)]
    #[case::disjoint(rect_polygon(20.0, 20.0, 24.0, 24.0), false)]
    fn constant_vs_constant_polygons(
        #[case] other: Polygon,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let other = geometry_constant(&Geometry::Polygon(other), 3)?;
        assert_contains(container, other, [expected; 3])
    }

    /// Partially overlapping polygons contain each other in neither direction.
    #[test]
    fn overlapping_polygons_contain_neither_way() -> VortexResult<()> {
        let a = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let b = geometry_constant(&Geometry::Polygon(rect_polygon(2.0, 2.0, 6.0, 6.0)), 2)?;
        assert_contains(a.clone(), b.clone(), [false; 2])?;
        assert_contains(b, a, [false; 2])
    }

    /// Containment is not symmetric: a polygon contains an interior point, but the point does
    /// not contain the polygon.
    #[test]
    fn contains_is_asymmetric() -> VortexResult<()> {
        let polygon = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 2)?;
        assert_contains(polygon.clone(), point.clone(), [true; 2])?;
        assert_contains(point, polygon, [false; 2])
    }

    /// Constant polygon vs point column: a strictly interior point is contained; points outside
    /// or exactly on the boundary are not (OGC contains excludes the boundary).
    #[test]
    fn constant_polygon_vs_point_column() -> VortexResult<()> {
        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let points = point_column(vec![2.0, 10.0, 0.0], vec![2.0, 10.0, 2.0])?;
        assert_contains(container, points, [true, false, false])
    }

    /// Polygon column vs constant point: only the polygon around the point contains it.
    #[test]
    fn polygon_column_vs_constant_point() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let around = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?,
            &mut ctx,
        )?;
        let away = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(20.0, 20.0, 24.0, 24.0)), 2)?,
            &mut ctx,
        )?;
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 2)?;

        assert_contains(around, point.clone(), [true; 2])?;
        assert_contains(away, point, [false; 2])
    }

    /// Constant container vs a linestring column: a row whose bounding rect pokes outside the
    /// container's rect is proven false by the rect pre-check alone; a fully inside row still
    /// needs (and passes) the exact test.
    #[test]
    fn constant_container_vs_row_rect_poking_outside() -> VortexResult<()> {
        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let lines = linestring_column(vec![
            vec![(1.0, 1.0), (3.0, 3.0)],
            vec![(1.0, 1.0), (9.0, 1.0)],
            vec![(5.0, 5.0), (9.0, 9.0)],
        ])?;
        assert_contains(container, lines, [true, false, false])
    }

    /// Column vs column pairs rows: each polygon row is tested against the point row at the
    /// same position.
    #[test]
    fn polygon_column_vs_point_column() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let polygons = materialize(
            geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?,
            &mut ctx,
        )?;
        let points = point_column(vec![2.0, 10.0], vec![2.0, 10.0])?;
        assert_contains(polygons, points, [true, false])
    }

    /// Output nullability mirrors the operands: nullable if any operand is nullable, otherwise
    /// non-nullable.
    #[test]
    fn output_nullability_mirrors_operands() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let non_nullable =
            SpatialContains.return_dtype(&EmptyOptions, &[dtype.clone(), dtype.clone()])?;
        assert!(!non_nullable.is_nullable());
        let nullable =
            SpatialContains.return_dtype(&EmptyOptions, &[dtype.as_nullable(), dtype])?;
        assert!(nullable.is_nullable());
        Ok(())
    }

    /// A null row in the contained operand yields a null verdict; valid rows keep their verdict
    /// (a strictly interior point is contained, an outside point is not).
    #[test]
    fn contains_propagates_null_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let points = nullable_point_column(vec![Some((2.0, 2.0)), None, Some((10.0, 10.0))])?;
        let contains = SpatialContains::try_new(container, points)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false]),
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// A constant-null operand produces an all-null output.
    #[test]
    fn contains_constant_null_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let null_const = ConstantArray::new(Scalar::null(point_dtype), 2).into_array();
        let points = point_column(vec![2.0, 10.0], vec![2.0, 10.0])?;
        let contains = SpatialContains::try_new(null_const, points)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// Both operands nullable columns: containment (asymmetric) is null wherever either the
    /// container or the contained row is null, and computed on the rows valid in both.
    #[test]
    fn contains_propagates_column_pair_nulls() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // A point contains another point only when they are equal.
        let container = nullable_point_column(vec![
            Some((1.0, 1.0)),
            None,
            Some((2.0, 2.0)),
            Some((3.0, 3.0)),
        ])?;
        let contained = nullable_point_column(vec![
            Some((1.0, 1.0)),
            Some((5.0, 5.0)),
            None,
            Some((4.0, 4.0)),
        ])?;
        let contains = SpatialContains::try_new(container, contained)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false, false]),
            Validity::from_iter([true, false, false, true]),
        )
        .into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// An entirely-null geometry column yields an all-null output.
    #[test]
    fn contains_all_null_column_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        let points = nullable_point_column(vec![None, None])?;
        let contains = SpatialContains::try_new(container, points)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// Two nullable columns whose nulls never line up: the combined mask is empty, so the output
    /// is all null.
    #[test]
    fn contains_column_pair_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let container = nullable_point_column(vec![Some((1.0, 1.0)), None])?;
        let contained = nullable_point_column(vec![None, Some((2.0, 2.0))])?;
        let contains = SpatialContains::try_new(container, contained)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    /// A non-geometry operand dtype is rejected up front, before execution.
    #[test]
    fn non_geometry_operand_is_rejected() -> VortexResult<()> {
        let spatial_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = SpatialContains.return_dtype(&EmptyOptions, &[spatial_dtype, numeric]);
        assert!(result.is_err());
        Ok(())
    }
}
