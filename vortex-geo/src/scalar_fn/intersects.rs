// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Intersects`: OGC intersection test between two native geometries.

use geo::Intersects;
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
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::validate_geometry_operands;
use crate::scalar_fn::execute::execute_null_propagating;

/// OGC `ST_Intersects` (not disjoint; boundary contact counts) between two native geometry
/// operands, each a column or a constant literal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoIntersects;

impl GeoIntersects {
    /// A lazy `ScalarFnArray` computing per-row whether operands `a` and `b` intersect; either may
    /// be constant. The output length is taken from `a`.
    pub fn try_new_array(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoIntersects, EmptyOptions).erased(),
            vec![a, b],
        )
    }
}

impl ScalarFnVTable for GeoIntersects {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.geo.intersects");
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
            _ => unreachable!("intersects has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_geometry_operands(dtypes)?;
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
        execute_null_propagating(&a, &b, |x, y| x.intersects(y), ctx)
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
}

#[cfg(test)]
mod tests {
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::MultiPolygon;
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
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use wkb::writer::WriteOptions;

    use super::GeoIntersects;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;

    /// A rectangle polygon with corners `(x0, y0)` and `(x1, y1)`, no holes.
    fn rect_polygon(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(
            LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
            vec![],
        )
    }

    /// The query region shared by the point tests: a `10x10` square with a `4..6` square hole.
    fn donut() -> Geometry {
        Geometry::Polygon(Polygon::new(
            LineString::from(vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
            vec![LineString::from(vec![
                (4.0, 4.0),
                (6.0, 4.0),
                (6.0, 6.0),
                (4.0, 6.0),
            ])],
        ))
    }

    /// Probes for [`donut`], one per verdict class: interior, exterior, on the outer boundary,
    /// and inside the hole.
    fn donut_probes() -> VortexResult<ArrayRef> {
        point_column(vec![2.0, 20.0, 0.0, 5.0], vec![2.0, 20.0, 5.0, 5.0])
    }

    /// [`donut_probes`]'s verdicts: interior and boundary contact intersect; exterior and
    /// in-hole do not.
    const DONUT_EXPECTED: [bool; 4] = [true, false, true, false];

    /// A constant column of length `len`, every row the native form of `geometry`.
    fn geometry_constant(geometry: &Geometry, len: usize) -> VortexResult<ArrayRef> {
        let mut buf = Vec::new();
        wkb::writer::write_geometry(&mut buf, geometry, &WriteOptions::default())
            .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;
        let scalar = crate::extension::native_geometry_scalar_from_wkb(&buf)?
            .ok_or_else(|| vortex_err!("unsupported geometry type"))?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    /// Materialize `array` so it is no longer a `Constant`, forcing the non-constant kernel paths.
    fn materialize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.execute::<Canonical>(ctx)?.into_array())
    }

    /// Execute `GeoIntersects(a, b)` and assert the per-row verdicts equal `expected`.
    fn assert_intersects(
        a: ArrayRef,
        b: ArrayRef,
        expected: impl IntoIterator<Item = bool>,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let intersects = GeoIntersects::try_new_array(a, b)?.into_array();
        assert_arrays_eq!(intersects, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    // The tests cover each `execute` dispatch arm in match order, then the edge cases.

    /// Constant vs constant: overlapping and touching (edge or corner) polygons intersect,
    /// disjoint ones do not; every output row carries the same verdict.
    #[rstest]
    #[case::overlapping(rect_polygon(2.0, 2.0, 6.0, 6.0), true)]
    #[case::touching_edge(rect_polygon(4.0, 0.0, 8.0, 4.0), true)]
    #[case::touching_corner(rect_polygon(4.0, 4.0, 8.0, 8.0), true)]
    #[case::disjoint(rect_polygon(20.0, 20.0, 24.0, 24.0), false)]
    fn constant_vs_constant_polygons(
        #[case] other: Polygon,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let a = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 3)?;
        let b = geometry_constant(&Geometry::Polygon(other), 3)?;
        assert_intersects(a, b, [expected; 3])
    }

    /// Point column vs constant polygon, with the constant on either side since intersection
    /// is symmetric: the [`DONUT_EXPECTED`] verdicts.
    #[test]
    fn point_column_vs_constant_polygon() -> VortexResult<()> {
        assert_intersects(
            donut_probes()?,
            geometry_constant(&donut(), 4)?,
            DONUT_EXPECTED,
        )?;
        assert_intersects(
            geometry_constant(&donut(), 4)?,
            donut_probes()?,
            DONUT_EXPECTED,
        )
    }

    /// Point column vs constant multipolygon: a point in either part intersects, a point in
    /// neither does not.
    #[test]
    fn point_column_vs_constant_multipolygon() -> VortexResult<()> {
        let query = Geometry::MultiPolygon(MultiPolygon::new(vec![
            rect_polygon(0.0, 0.0, 2.0, 2.0),
            rect_polygon(10.0, 10.0, 12.0, 12.0),
        ]));
        let points = point_column(vec![1.0, 11.0, 5.0], vec![1.0, 11.0, 5.0])?;
        assert_intersects(points, geometry_constant(&query, 3)?, [true, true, false])
    }

    /// Polygon column vs constant: the same pairs and verdicts as
    /// [`constant_vs_constant_polygons`].
    #[rstest]
    #[case::overlapping(rect_polygon(2.0, 2.0, 6.0, 6.0), true)]
    #[case::touching_edge(rect_polygon(4.0, 0.0, 8.0, 4.0), true)]
    #[case::disjoint(rect_polygon(20.0, 20.0, 24.0, 24.0), false)]
    fn polygon_column_vs_constant(
        #[case] other: Polygon,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let column = materialize(geometry_constant(&Geometry::Polygon(other), 2)?, &mut ctx)?;
        let query = geometry_constant(&Geometry::Polygon(rect_polygon(0.0, 0.0, 4.0, 4.0)), 2)?;
        assert_intersects(column, query, [expected; 2])
    }

    /// Point column vs point column: rows intersect exactly at equal coordinates.
    #[test]
    fn point_column_vs_point_column() -> VortexResult<()> {
        let a = point_column(vec![0.0, 1.0], vec![0.0, 1.0])?;
        let b = point_column(vec![0.0, 2.0], vec![0.0, 1.0])?;
        assert_intersects(a, b, [true, false])
    }

    /// Point column vs polygon column pairwise: agrees with point column vs constant on the
    /// same data.
    #[test]
    fn columns_agree_with_constant() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let constant = geometry_constant(&donut(), 4)?;
        let column = materialize(constant.clone(), &mut ctx)?;

        let against_constant =
            GeoIntersects::try_new_array(donut_probes()?, constant)?.into_array();
        let pairwise = GeoIntersects::try_new_array(donut_probes()?, column)?.into_array();

        assert_arrays_eq!(against_constant, pairwise, &mut ctx);
        Ok(())
    }

    /// An empty constant geometry intersects nothing.
    #[test]
    fn empty_constant_intersects_nothing() -> VortexResult<()> {
        let empty = Geometry::LineString(LineString::new(Vec::<Coord>::new()));
        let points = point_column(vec![0.0, 1.0], vec![0.0, 1.0])?;
        assert_intersects(points, geometry_constant(&empty, 2)?, [false, false])
    }

    /// Output nullability mirrors the operands: nullable if any operand is nullable, otherwise
    /// non-nullable.
    #[test]
    fn output_nullability_mirrors_operands() -> VortexResult<()> {
        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let non_nullable =
            GeoIntersects.return_dtype(&EmptyOptions, &[dtype.clone(), dtype.clone()])?;
        assert!(!non_nullable.is_nullable());
        let nullable = GeoIntersects.return_dtype(&EmptyOptions, &[dtype.as_nullable(), dtype])?;
        assert!(nullable.is_nullable());
        Ok(())
    }

    /// A null row in a geometry operand yields a null verdict; valid rows keep their verdict
    /// (interior intersects, exterior does not).
    #[test]
    fn intersects_propagates_null_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = nullable_point_column(vec![Some((2.0, 2.0)), None, Some((20.0, 20.0))])?;
        let query = geometry_constant(&donut(), 3)?;
        let intersects = GeoIntersects::try_new_array(points, query)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false]),
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_arrays_eq!(intersects, expected, &mut ctx);
        Ok(())
    }

    /// A constant-null operand produces an all-null output.
    #[test]
    fn intersects_constant_null_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let null_const = ConstantArray::new(Scalar::null(point_dtype), 2).into_array();
        let points = point_column(vec![2.0, 20.0], vec![2.0, 20.0])?;
        let intersects = GeoIntersects::try_new_array(null_const, points)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(intersects, expected, &mut ctx);
        Ok(())
    }

    /// Both operands nullable columns: the verdict is null wherever either row is null, and
    /// computed on the rows valid in both (points intersect exactly when equal).
    #[test]
    fn intersects_propagates_column_pair_nulls() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![
            Some((0.0, 0.0)),
            None,
            Some((2.0, 2.0)),
            Some((3.0, 3.0)),
        ])?;
        let b = nullable_point_column(vec![
            Some((0.0, 0.0)),
            Some((5.0, 5.0)),
            None,
            Some((9.0, 9.0)),
        ])?;
        let intersects = GeoIntersects::try_new_array(a, b)?.into_array();

        let expected = BoolArray::new(
            BitBuffer::from_iter([true, false, false, false]),
            Validity::from_iter([true, false, false, true]),
        )
        .into_array();
        assert_arrays_eq!(intersects, expected, &mut ctx);
        Ok(())
    }

    /// An entirely-null geometry column yields an all-null output.
    #[test]
    fn intersects_all_null_column_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = nullable_point_column(vec![None, None])?;
        let query = geometry_constant(&donut(), 2)?;
        let intersects = GeoIntersects::try_new_array(points, query)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(intersects, expected, &mut ctx);
        Ok(())
    }

    /// Two nullable columns whose nulls never line up: the combined mask is empty, so the output
    /// is all null.
    #[test]
    fn intersects_column_pair_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = nullable_point_column(vec![Some((0.0, 0.0)), None])?;
        let b = nullable_point_column(vec![None, Some((1.0, 1.0))])?;
        let intersects = GeoIntersects::try_new_array(a, b)?.into_array();

        let expected =
            BoolArray::new(BitBuffer::from_iter([false, false]), Validity::AllInvalid).into_array();
        assert_arrays_eq!(intersects, expected, &mut ctx);
        Ok(())
    }

    /// A non-geometry operand dtype is rejected up front, before execution.
    #[test]
    fn non_geometry_operand_is_rejected() -> VortexResult<()> {
        let geo = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let numeric = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = GeoIntersects.return_dtype(&EmptyOptions, &[geo, numeric]);
        assert!(result.is_err());
        Ok(())
    }
}
