// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Straight-line (Euclidean) distance between geometries; "planar" distance in GIS terms.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::extension::Geometry;
use crate::extension::GeometryKind;
use crate::extension::geometry::coordinate::coordinate_from_scalar;
use crate::extension::geometry::coordinate::parse_storage;

/// Straight-line (L2) distance between `(ax, ay)` and `(bx, by)`.
fn euclidean_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// Planar (Euclidean) distance between two geometry columns, like PostGIS `ST_Distance`;
/// `z`/`m` are ignored.
///
/// Operands are type-checked at construction: point kind (the only kernel so far),
/// non-nullable, and sharing a CRS. Execution dispatches on the operands' [`GeometryKind`]s.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoDistance;

impl GeoDistance {
    /// A lazy `ScalarFnArray` of per-row distances between `a` and `b`, with `a`'s length.
    pub fn try_new_array(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        let len = a.len();
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoDistance, EmptyOptions).erased(),
            vec![a, b],
            len,
        )
    }
}

impl ScalarFnVTable for GeoDistance {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.geo.distance")
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
            _ => unreachable!("distance has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        for dtype in arg_dtypes {
            let kind = Geometry::kind_of(dtype)?;
            vortex_ensure!(
                kind == GeometryKind::Point,
                "distance over {kind} geometries is not yet implemented"
            );
            vortex_ensure!(
                !dtype.is_nullable(),
                "distance over nullable geometry is not yet supported, was {dtype}"
            );
        }
        if let [a, b] = arg_dtypes {
            let a_crs = &a.as_extension().metadata::<Geometry>().crs;
            let b_crs = &b.as_extension().metadata::<Geometry>().crs;
            vortex_ensure!(
                a_crs == b_crs,
                "distance operands must share a CRS, was {} and {}",
                a_crs.as_deref().unwrap_or("unreferenced"),
                b_crs.as_deref().unwrap_or("unreferenced")
            );
        }
        Ok(DType::Primitive(PType::F64, Nullability::NonNullable))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let a = args.get(0)?;
        let b = args.get(1)?;
        match (Geometry::kind_of(a.dtype())?, Geometry::kind_of(b.dtype())?) {
            (GeometryKind::Point, GeometryKind::Point) => point_to_point_distance(&a, &b, ctx),
            (a_kind, b_kind) => {
                vortex_bail!("distance({a_kind}, {b_kind}) is not yet implemented")
            }
        }
    }
}

/// Per-row distance between two point columns; either operand (or both) may be constant.
fn point_to_point_distance(
    a: &ArrayRef,
    b: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match (a.as_opt::<Constant>(), b.as_opt::<Constant>()) {
        (Some(qa), Some(qb)) => {
            let qa = coordinate_from_scalar(qa.scalar())?;
            let qb = coordinate_from_scalar(qb.scalar())?;
            let distance = euclidean_distance(qa.x, qa.y, qb.x, qb.y);
            Ok(ConstantArray::new(
                Scalar::primitive(distance, Nullability::NonNullable),
                a.len(),
            )
            .into_array())
        }
        (Some(query), None) => point_to_constant_distance(b, query.scalar(), ctx),
        (None, Some(query)) => point_to_constant_distance(a, query.scalar(), ctx),
        (None, None) => {
            let a_coords = parse_storage(a, ctx)?;
            let b_coords = parse_storage(b, ctx)?;
            let distances = a_coords
                .xs
                .as_slice::<f64>()
                .iter()
                .zip(a_coords.ys.as_slice::<f64>())
                .zip(
                    b_coords
                        .xs
                        .as_slice::<f64>()
                        .iter()
                        .zip(b_coords.ys.as_slice::<f64>()),
                )
                .map(|((&ax, &ay), (&bx, &by))| euclidean_distance(ax, ay, bx, by));
            Ok(PrimitiveArray::from_iter(distances).into_array())
        }
    }
}

/// Per-row distance from `points` to the constant `query` point, decoded once. Distance is
/// symmetric, so this serves a constant on either side.
fn point_to_constant_distance(
    points: &ArrayRef,
    query: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let query = coordinate_from_scalar(query)?;
    let coords = parse_storage(points, ctx)?;
    let distances = coords
        .xs
        .as_slice::<f64>()
        .iter()
        .zip(coords.ys.as_slice::<f64>())
        .map(|(&x, &y)| euclidean_distance(x, y, query.x, query.y));
    Ok(PrimitiveArray::from_iter(distances).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::GeoDistance;
    use super::euclidean_distance;
    use crate::extension::Geometry;
    use crate::extension::GeometryKind;

    /// A point `Geometry` column with the given CRS over the given x/y coordinates.
    fn point_column_with_crs(
        xs: Vec<f64>,
        ys: Vec<f64>,
        crs: Option<String>,
    ) -> VortexResult<ArrayRef> {
        let storage = StructArray::from_fields(&[
            ("x", PrimitiveArray::from_iter(xs).into_array()),
            ("y", PrimitiveArray::from_iter(ys).into_array()),
        ])?
        .into_array();
        let dtype = Geometry::dtype(GeometryKind::Point, crs, storage.dtype().clone())?;
        Ok(ExtensionArray::new(dtype.erased(), storage).into_array())
    }

    /// A point `Geometry` column (CRS `EPSG:4326`) over the given x/y coordinates.
    fn point_column(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
        point_column_with_crs(xs, ys, Some("EPSG:4326".to_string()))
    }

    /// A constant point column of length `len`, every row at `(x, y)`.
    fn point_constant(
        x: f64,
        y: f64,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let single = point_column(vec![x], vec![y])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(single, len).into_array())
    }

    /// Execute a `GeoDistance` array and read back its per-row `f64` distances.
    fn distances(distance: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<f64>> {
        Ok(distance
            .execute::<Canonical>(ctx)?
            .into_primitive()
            .as_slice::<f64>()
            .to_vec())
    }

    /// The kernel computes straight-line distance (the 3–4–5 triangle).
    #[test]
    fn euclidean_distance_is_straight_line() {
        assert_eq!(euclidean_distance(0.0, 0.0, 3.0, 4.0), 5.0);
        assert_eq!(euclidean_distance(1.5, -1.5, 1.5, -1.5), 0.0);
    }

    /// Per-row distance between a point column and a constant query point.
    #[test]
    fn distance_over_points() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![0.0, 3.0, 0.0, 3.0], vec![0.0, 0.0, 4.0, 4.0])?;
        let b = point_constant(0.0, 0.0, 4, &mut ctx)?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![0.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    /// Column-to-column distance pairs corresponding rows of the two columns.
    #[test]
    fn distance_between_columns() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![0.0, 1.0], vec![0.0, 1.0])?;
        let b = point_column(vec![3.0, 1.0], vec![4.0, 1.0])?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![5.0, 0.0]);
        Ok(())
    }

    /// The constant query point may be either operand; distance is symmetric.
    #[test]
    fn distance_with_constant_first_operand() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let a = point_constant(0.0, 0.0, 4, &mut ctx)?;
        let b = point_column(vec![0.0, 3.0, 0.0, 3.0], vec![0.0, 0.0, 4.0, 4.0])?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![0.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    /// Two constant operands: every row has the same distance.
    #[test]
    fn distance_between_two_constants() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let a = point_constant(0.0, 0.0, 3, &mut ctx)?;
        let b = point_constant(3.0, 4.0, 3, &mut ctx)?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![5.0, 5.0, 5.0]);
        Ok(())
    }

    /// Distance is signed `(geometry, geometry)`: a non-geometry operand is rejected at
    /// construction.
    #[test]
    fn distance_rejects_non_geometry_operands() -> VortexResult<()> {
        let a = point_column(vec![0.0], vec![0.0])?;
        let b = PrimitiveArray::from_iter(vec![1.0f64]).into_array();
        assert!(GeoDistance::try_new_array(a, b).is_err());
        Ok(())
    }

    /// Operands in different (or missing vs. set) CRS are rejected at construction: their raw
    /// ordinates are not comparable.
    #[test]
    fn distance_rejects_mismatched_crs() -> VortexResult<()> {
        let a = point_column(vec![0.0], vec![0.0])?;
        let b = point_column_with_crs(vec![1.0], vec![1.0], Some("EPSG:3857".to_string()))?;
        assert!(GeoDistance::try_new_array(a, b).is_err());

        let a = point_column(vec![0.0], vec![0.0])?;
        let unreferenced = point_column_with_crs(vec![1.0], vec![1.0], None)?;
        assert!(GeoDistance::try_new_array(a, unreferenced).is_err());
        Ok(())
    }

    /// Nullable point columns are rejected at construction until validity propagation exists.
    #[test]
    fn distance_rejects_nullable_points() -> VortexResult<()> {
        let storage = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                PrimitiveArray::from_iter(vec![1.0f64]).into_array(),
                PrimitiveArray::from_iter(vec![2.0f64]).into_array(),
            ],
            1,
            Validity::AllValid,
        )?
        .into_array();
        let dtype = Geometry::dtype(
            GeometryKind::Point,
            Some("EPSG:4326".to_string()),
            storage.dtype().clone(),
        )?;
        let nullable = ExtensionArray::new(dtype.erased(), storage).into_array();

        let b = point_column(vec![2.0], vec![2.0])?;
        assert!(GeoDistance::try_new_array(nullable, b).is_err());
        Ok(())
    }
}
