// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Contains`: OGC containment test between two native geometries.

use std::cell::OnceCell;

use geo::BoundingRect;
use geo::Contains;
use geo::PreparedGeometry;
use geo::Relate;
use geo_types::Geometry;
use geo_types::Rect;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::unstable::row::InitializedElement;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_array::scalar_fn::unstable::row::UninitElementSink;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::scalar_fn::row::GeometryRow;
#[cfg(test)]
use crate::scalar_fn::row::probe;

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

impl RowFn for SpatialContains {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["a", "b"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.contains");
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

    /// Containment is not symmetric, so `a` is always the container and `b` the contained.
    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_prepared_into::<(GeometryRow, GeometryRow), UninitElementSink<bool>, _, _>(
            |(a, b)| {
                #[cfg(test)]
                probe::record(a.is_some(), b.is_some());
                ConstOperands {
                    a: a.map(PreparedOperand::new),
                    b: b.map(PreparedOperand::new),
                }
            },
            |operands, (a, b), output| {
                // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
                unsafe { InitializedElement::write(output, contains_row_prepared(operands, a, b)) }
            },
        )
    }
}

/// Per-batch state for the contains row kernel: the prepared form of whichever operand is
/// constant for the batch. `None` marks an operand that varies by row.
struct ConstOperands {
    /// Operand `a` (the container) when it is batch-constant.
    a: Option<PreparedOperand>,

    /// Operand `b` (the contained) when it is batch-constant.
    b: Option<PreparedOperand>,
}

/// One batch-constant operand: its bounding rectangle and the [`PreparedGeometry`] built on the
/// first row whose pairing routes through relate.
///
/// The build is lazy because preparation (self-noding the topology graph plus an R*-tree over the
/// edges) costs `O(edges log edges)` and pays off only on relate-routed pairings; a batch of
/// point rows against a constant polygon never touches it, and preparing a large constant eagerly
/// would charge such a batch for nothing.
struct PreparedOperand {
    /// The constant's bounding rectangle, folded once for conservative row rejection.
    bbox: Option<Rect<f64>>,

    /// The constant's prepared form, initialized only when a relate route needs it.
    prepared: OnceCell<PreparedGeometry<'static, Geometry<f64>, f64>>,
}

impl PreparedOperand {
    fn new(geometry: &Geometry<f64>) -> Self {
        Self {
            bbox: finite_bounding_rect(geometry),
            prepared: OnceCell::new(),
        }
    }

    /// Return the prepared geometry, cloning the decoded constant only on first use.
    ///
    /// `geometry` **must** be the constant represented by this state. The row kernel maintains
    /// that relationship by passing the operand from the same decoded constant column that
    /// produced this [`PreparedOperand`].
    fn get(&self, geometry: &Geometry<f64>) -> &PreparedGeometry<'static, Geometry<f64>, f64> {
        self.prepared
            .get_or_init(|| PreparedGeometry::from(geometry.clone()))
    }
}

/// Returns a bounding rectangle only when ordered comparisons can conservatively reject a row.
///
/// Geo permits non-finite coordinates. A rectangle containing NaN cannot prove non-containment,
/// because its ordered comparisons can return false even when the exact algorithm accepts the
/// geometry.
fn finite_bounding_rect(geometry: &Geometry<f64>) -> Option<Rect<f64>> {
    let bbox = geometry.bounding_rect()?;
    let min = bbox.min();
    let max = bbox.max();

    [min.x, min.y, max.x, max.y]
        .into_iter()
        .all(f64::is_finite)
        .then_some(bbox)
}

/// How geo's `a.contains(b)` computes its verdict for a pairing.
enum ContainsRoute {
    /// `a.relate(b).is_contains()`.
    ForwardRelate,

    /// `b.relate(a).is_within()`, how geo phrases relate for `MultiPolygon` containers.
    ReversedRelate,

    /// A direct algorithm (coordinate position, point arithmetic); nothing to prepare.
    Direct,
}

/// The route geo 0.31's `Contains` dispatch takes for `a.contains(b)`.
///
/// The prepared substitution in [`contains_row_prepared`] **must** run relate exactly where geo
/// runs relate, with the same argument order, because geo's direct algorithms are not everywhere
/// bit-identical to a relate matrix query (they resolve degenerate and boundary cases with
/// different arithmetic). The relate rows below transcribe geo's `impl_contains_from_relate!`
/// lists per container type; everything else, notably every `Point`/`MultiPoint` contained side
/// and every `Point` container, is direct.
///
/// **This table is coupled to the geo version.** It transcribes a dispatch that geo is free to
/// reshuffle in any release, and a wrong row is a silently wrong verdict rather than a build error.
/// The workspace therefore pins `geo = "=0.31.0"`: taking any new geo, patch releases included, is
/// a deliberate edit of that line, and the edit must re-verify this table against
/// `impl_contains_from_relate!`.
///
/// `constant_operands_agree_with_columns` is the mechanical check, and it is **not** complete: it
/// compares the prepared route against plain `a.contains(b)` only for the container types it has
/// cases for. `routes_agree_with_geo_for_every_container` covers the rest, one representative
/// pairing per container variant, and is the one to extend when geo grows a geometry type. Both
/// stay green wherever relate and the direct algorithm agree, so neither replaces the pin.
fn contains_route(a: &Geometry<f64>, b: &Geometry<f64>) -> ContainsRoute {
    use Geometry as G;

    match (a, b) {
        // Line contains [Polygon, MultiLineString, MultiPolygon, GeometryCollection, Rect,
        // Triangle].
        (
            G::Line(_),
            G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // LineString contains [Polygon, MultiPoint, MultiLineString, MultiPolygon,
        // GeometryCollection, Rect, Triangle].
        | (
            G::LineString(_),
            G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // MultiLineString contains everything except Point.
        | (
            G::MultiLineString(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // MultiPoint contains [Line, LineString, Polygon, MultiLineString, MultiPolygon,
        // GeometryCollection, Rect, Triangle].
        | (
            G::MultiPoint(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // Polygon contains everything except Point and MultiPoint.
        | (
            G::Polygon(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        )
        // Rect contains [Line, LineString, MultiPoint, MultiLineString, MultiPolygon,
        // GeometryCollection, Triangle]; Rect contains Rect and Polygon are direct.
        | (
            G::Rect(_),
            G::Line(_)
            | G::LineString(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Triangle(_),
        )
        // Triangle and GeometryCollection contain everything except Point.
        | (
            G::Triangle(_) | G::GeometryCollection(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiPoint(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        ) => ContainsRoute::ForwardRelate,

        // MultiPolygon contains everything except Point and MultiPoint, phrased reversed.
        (
            G::MultiPolygon(_),
            G::Line(_)
            | G::LineString(_)
            | G::Polygon(_)
            | G::MultiLineString(_)
            | G::MultiPolygon(_)
            | G::GeometryCollection(_)
            | G::Rect(_)
            | G::Triangle(_),
        ) => ContainsRoute::ReversedRelate,

        _ => ContainsRoute::Direct,
    }
}

/// Computes one row of contains, substituting a prepared graph for a constant operand on the
/// pairings geo itself answers through relate.
///
/// [`PreparedGeometry`] carries the operand's self-noded topology graph and edge R*-tree, so a
/// relate against it skips rebuilding both and reads its bounding rect from cache; geo asserts
/// the cached graph equal to a freshly built one (its `swap_arg_index` test), which is what makes
/// the substitution result-preserving. Before dispatch, a disjoint constant-side bounding rect
/// conservatively rejects the row, matching the columnar implementation's #9076 optimization.
/// All other rows delegate to the same direct or relate route as `a.contains(b)`.
fn contains_row_prepared(operands: &ConstOperands, a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    let rejected = match (&operands.a, &operands.b) {
        (None, None) => false,
        (Some(const_a), Some(const_b)) => const_a
            .bbox
            .zip(const_b.bbox)
            .is_some_and(|(bbox_a, bbox_b)| !bbox_a.contains(&bbox_b)),
        (Some(const_a), None) => const_a
            .bbox
            .zip(finite_bounding_rect(b))
            .is_some_and(|(bbox_a, bbox_b)| !bbox_a.contains(&bbox_b)),
        (None, Some(const_b)) => finite_bounding_rect(a)
            .zip(const_b.bbox)
            .is_some_and(|(bbox_a, bbox_b)| !bbox_a.contains(&bbox_b)),
    };

    if rejected {
        return false;
    }

    match contains_route(a, b) {
        ContainsRoute::Direct => a.contains(b),
        ContainsRoute::ForwardRelate => match (&operands.a, &operands.b) {
            (Some(const_a), Some(const_b)) => const_a.get(a).relate(const_b.get(b)).is_contains(),
            (Some(const_a), None) => const_a.get(a).relate(b).is_contains(),
            (None, Some(const_b)) => a.relate(const_b.get(b)).is_contains(),
            (None, None) => a.contains(b),
        },
        ContainsRoute::ReversedRelate => match (&operands.a, &operands.b) {
            (Some(const_a), Some(const_b)) => const_b.get(b).relate(const_a.get(a)).is_within(),
            (Some(const_a), None) => b.relate(const_a.get(a)).is_within(),
            (None, Some(const_b)) => const_b.get(b).relate(a).is_within(),
            (None, None) => a.contains(b),
        },
    }
}

#[cfg(test)]
mod tests {
    use geo::Contains;
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::GeometryCollection;
    use geo_types::Line;
    use geo_types::LineString;
    use geo_types::MultiLineString;
    use geo_types::MultiPoint;
    use geo_types::MultiPolygon;
    use geo_types::Point;
    use geo_types::Polygon;
    use geo_types::Rect;
    use geo_types::Triangle;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
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

    use super::ConstOperands;
    use super::PreparedOperand;
    use super::SpatialContains;
    use super::contains_row_prepared;
    use crate::scalar_fn::row::probe::assert_prepared_agrees_with_columns;
    use crate::test_harness::linestring_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

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

    /// A non-finite bounding rectangle cannot reject a containment that the exact geometry
    /// algorithm accepts.
    #[test]
    fn nan_bounding_rect_does_not_reject_containment() {
        let container = multipoint(vec![(f64::NAN, f64::NAN), (1.0, 1.0)]);
        let contained = point(1.0, 1.0);
        let operands = ConstOperands {
            a: Some(PreparedOperand::new(&container)),
            b: Some(PreparedOperand::new(&contained)),
        };

        assert!(container.contains(&contained));
        assert!(contains_row_prepared(&operands, &container, &contained));
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

    /// Constant container vs a linestring column: a row whose bounding rect pokes outside the
    /// container's is not contained, while one wholly inside is. Carried over from the columnar
    /// bounding-rect rejection in #9076, since it constrains the verdict rather than the mechanism.
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

    /// A nullable polygon column: unit squares at `centers`, the rows where `nulls` is true
    /// masked out, spelled as `Masked` over non-nullable storage.
    fn nullable_squares(centers: &[(f64, f64)], nulls: &[bool]) -> VortexResult<ArrayRef> {
        let squares = centers
            .iter()
            .map(|&(x, y)| {
                vec![vec![
                    (x - 1.0, y - 1.0),
                    (x + 1.0, y - 1.0),
                    (x + 1.0, y + 1.0),
                    (x - 1.0, y + 1.0),
                    (x - 1.0, y - 1.0),
                ]]
            })
            .collect();
        let polygons = polygon_column(squares)?;

        Ok(
            MaskedArray::try_new(polygons, Validity::from_iter(nulls.iter().map(|n| !n)))?
                .into_array(),
        )
    }

    /// Nullable geometry operands conjoin their validity before computing containment.
    #[test]
    fn contains_nullable_geometries_conjoins_validity() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let centers = [(0.0, 0.0), (5.0, 5.0), (0.5, -0.2), (9.0, 9.0), (0.0, 1.0)];
        let nulls = [false, true, false, false, true];
        let polygons = nullable_squares(&centers, &nulls)?;
        let points = nullable_point_column(vec![
            Some((0.0, 0.0)),
            Some((5.0, 5.0)),
            None,
            Some((0.0, 0.0)),
            Some((0.0, 1.0)),
        ])?;

        let actual = SpatialContains::try_new(polygons, points)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        let expected = BoolArray::from_iter([Some(true), None, None, Some(false), None]);

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Geometry types without a null-tolerant decode fall back to filtering valid rows.
    #[test]
    fn contains_unsupported_geometry_falls_back_to_filter() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let validity = Validity::from_iter([true, false, true, true]);
        let lines = linestring_column(vec![
            vec![(0.0, 0.0), (4.0, 4.0)],
            vec![(0.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 3.0)],
            vec![(0.0, 4.0), (4.0, 0.0)],
        ])?;
        let nullable_lines = MaskedArray::try_new(lines.clone(), validity.clone())?.into_array();
        let point = geometry_constant(&Geometry::Point(Point::new(2.0, 2.0)), 4)?;

        let expected = SpatialContains::try_new(lines, point.clone())?.into_array();
        let expected = MaskedArray::try_new(expected, validity)?.into_array();
        let actual = SpatialContains::try_new(nullable_lines, point)?
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_arrays_eq!(actual, expected, &mut ctx);
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

    // The prepared-vs-expanded agreement grid: every constant arrangement of a pairing must
    // return exactly what the fully expanded columns return.

    /// A point geometry.
    fn point(x: f64, y: f64) -> Geometry {
        Geometry::Point(Point::new(x, y))
    }

    /// A linestring geometry through `coords`.
    fn line(coords: Vec<(f64, f64)>) -> Geometry {
        Geometry::LineString(LineString::from(coords))
    }

    /// A multipoint geometry over `coords`.
    fn multipoint(coords: Vec<(f64, f64)>) -> Geometry {
        Geometry::MultiPoint(MultiPoint::from(coords))
    }

    /// A two-point line segment geometry, the `Line` container variant.
    fn line_geometry(start: (f64, f64), end: (f64, f64)) -> Geometry {
        Geometry::Line(Line::new(
            Coord {
                x: start.0,
                y: start.1,
            },
            Coord { x: end.0, y: end.1 },
        ))
    }

    /// A multilinestring geometry over one linestring per entry of `parts`.
    fn multilinestring(parts: Vec<Vec<(f64, f64)>>) -> Geometry {
        Geometry::MultiLineString(MultiLineString::new(
            parts.into_iter().map(LineString::from).collect(),
        ))
    }

    /// A geometry collection wrapping `parts`.
    fn collection(parts: Vec<Geometry>) -> Geometry {
        Geometry::GeometryCollection(GeometryCollection::from(parts))
    }

    /// An axis-aligned rectangle geometry, the `Rect` container variant.
    fn rect_geometry(x0: f64, y0: f64, x1: f64, y1: f64) -> Geometry {
        Geometry::Rect(Rect::new(Coord { x: x0, y: y0 }, Coord { x: x1, y: y1 }))
    }

    /// A triangle geometry large enough to contain the small test polygons.
    fn triangle_geometry() -> Geometry {
        Geometry::Triangle(Triangle::new(
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 8.0, y: 0.0 },
            Coord { x: 0.0, y: 8.0 },
        ))
    }

    /// A two-part multipolygon: `4x4` squares at the origin and at `(10, 10)`.
    fn two_part_multipolygon() -> Geometry {
        Geometry::MultiPolygon(MultiPolygon::new(vec![
            rect_polygon(0.0, 0.0, 4.0, 4.0),
            rect_polygon(10.0, 10.0, 14.0, 14.0),
        ]))
    }

    /// Every container variant `contains_route` distinguishes, checked against plain
    /// `a.contains(b)` in all four constant arrangements.
    ///
    /// Every case is a containment geo answers `true`, which the test asserts: a pairing that is
    /// false regardless of route (a lower-dimensional container, say) also agrees regardless of
    /// route, and pins nothing. A true case fails when the prepared substitution diverges from
    /// geo: a table row whose relate phrasing disagrees with geo's dispatch on this input, or a
    /// bounding-rect prescreen that wrongly rejects a contained row. It is **not** a version
    /// tripwire: a geo release that reshuffles its dispatch stays green wherever relate and the
    /// direct algorithm agree, which is why the workspace pins `geo` exactly.
    ///
    /// This is the table's own regression, and the one to extend when geo grows a geometry type:
    /// `constant_operands_agree_with_columns` below goes through real arrays and so is the better
    /// end-to-end check, but it only covers the container types it has cases for, and WKB decoding
    /// limits which types those can be. The MultiPoint and Line containers route relate only for
    /// contained types a MultiPoint or Line can rarely contain, so their true cases lean on
    /// `GeometryCollection` membership and collinear `MultiLineString` parts respectively.
    #[rstest]
    #[case::point(point(1.0, 1.0), point(1.0, 1.0))]
    #[case::line(line_geometry((0.0, 0.0), (4.0, 4.0)), point(2.0, 2.0))]
    #[case::line_x_multilinestring(line_geometry((0.0, 0.0), (4.0, 4.0)), multilinestring(vec![vec![(1.0, 1.0), (2.0, 2.0)]]))]
    #[case::linestring(line(vec![(0.0, 0.0), (4.0, 4.0)]), multipoint(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::polygon(rect_polygon(0.0, 0.0, 8.0, 8.0).into(), rect_polygon(2.0, 2.0, 4.0, 4.0).into())]
    #[case::multipoint(multipoint(vec![(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]), collection(vec![point(2.0, 2.0)]))]
    #[case::multilinestring(multilinestring(vec![vec![(0.0, 0.0), (4.0, 4.0)]]), line(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::multipolygon(two_part_multipolygon(), rect_polygon(1.0, 1.0, 3.0, 3.0).into())]
    #[case::geometrycollection(collection(vec![rect_polygon(0.0, 0.0, 8.0, 8.0).into()]), rect_polygon(2.0, 2.0, 4.0, 4.0).into())]
    #[case::rect(rect_geometry(0.0, 0.0, 8.0, 8.0), line(vec![(2.0, 2.0), (4.0, 4.0)]))]
    #[case::triangle(triangle_geometry(), rect_polygon(1.0, 1.0, 2.0, 2.0).into())]
    fn routes_agree_with_geo_for_every_container(#[case] a: Geometry, #[case] b: Geometry) {
        let expected = a.contains(&b);
        assert!(
            expected,
            "route cases must be containments geo answers true, or every route agrees vacuously",
        );

        let arrangements = [
            (None, None),
            (Some(PreparedOperand::new(&a)), None),
            (None, Some(PreparedOperand::new(&b))),
            (
                Some(PreparedOperand::new(&a)),
                Some(PreparedOperand::new(&b)),
            ),
        ];

        for (index, (const_a, const_b)) in arrangements.into_iter().enumerate() {
            let operands = ConstOperands {
                a: const_a,
                b: const_b,
            };
            assert_eq!(
                contains_row_prepared(&operands, &a, &b),
                expected,
                "arrangement {index} disagrees with geo's own contains",
            );
        }
    }

    /// Constant arrangements agree with expanded columns across the routes the prepared kernel
    /// distinguishes: forward relate (polygon, linestring and multipoint containers), reversed
    /// relate (multipolygon containers), and the direct pairings (a point on either side,
    /// multipoint over multipoint, polygon over multipoint), including boundary contact,
    /// crossing, disjoint and empty cases.
    #[rstest]
    #[case::polygon_nested_polygon(rect_polygon(0.0, 0.0, 8.0, 8.0).into(), rect_polygon(2.0, 2.0, 4.0, 4.0).into())]
    #[case::polygon_touching_from_inside(rect_polygon(0.0, 0.0, 8.0, 8.0).into(), rect_polygon(0.0, 2.0, 2.0, 4.0).into())]
    #[case::polygon_overlapping_polygon(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), rect_polygon(2.0, 2.0, 6.0, 6.0).into())]
    #[case::polygon_disjoint_polygon(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), rect_polygon(20.0, 20.0, 24.0, 24.0).into())]
    #[case::polygon_x_point_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(2.0, 2.0))]
    #[case::polygon_x_point_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(0.0, 2.0))]
    #[case::polygon_x_point_outside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(20.0, 20.0))]
    #[case::polygon_x_nan_point(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), point(f64::NAN, 2.0))]
    #[case::polygon_x_linestring_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::polygon_x_linestring_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(0.0, 1.0), (0.0, 3.0)]))]
    #[case::polygon_x_linestring_crossing(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![(-2.0, 2.0), (2.0, 2.0)]))]
    #[case::polygon_x_empty_linestring(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), line(vec![]))]
    #[case::polygon_x_multipoint_inside(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), multipoint(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::polygon_x_multipoint_on_boundary(rect_polygon(0.0, 0.0, 4.0, 4.0).into(), multipoint(vec![(0.0, 1.0), (0.0, 3.0)]))]
    #[case::linestring_x_multipoint_on_line(line(vec![(0.0, 0.0), (4.0, 4.0)]), multipoint(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::multipoint_x_multipoint_subset(multipoint(vec![(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]), multipoint(vec![(2.0, 2.0)]))]
    #[case::multipoint_x_linestring_between_points(multipoint(vec![(0.0, 0.0), (4.0, 4.0)]), line(vec![(1.0, 1.0), (2.0, 2.0)]))]
    #[case::multipolygon_x_polygon_in_one_part(two_part_multipolygon(), rect_polygon(1.0, 1.0, 3.0, 3.0).into())]
    #[case::multipolygon_x_polygon_straddling(two_part_multipolygon(), rect_polygon(3.0, 3.0, 11.0, 11.0).into())]
    #[case::multipolygon_x_polygon_disjoint(two_part_multipolygon(), rect_polygon(20.0, 20.0, 24.0, 24.0).into())]
    #[case::multipolygon_x_point_inside(two_part_multipolygon(), point(11.0, 11.0))]
    #[case::point_x_point_equal(point(1.0, 1.0), point(1.0, 1.0))]
    #[case::point_x_polygon(point(2.0, 2.0), rect_polygon(0.0, 0.0, 4.0, 4.0).into())]
    fn constant_operands_agree_with_columns(
        #[case] a: Geometry,
        #[case] b: Geometry,
    ) -> VortexResult<()> {
        assert_prepared_agrees_with_columns(
            SpatialContains::try_new,
            geometry_constant(&a, 3)?,
            geometry_constant(&b, 3)?,
        )
    }
}
