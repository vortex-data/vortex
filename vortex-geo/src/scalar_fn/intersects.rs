// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Intersects`: whether two native geometries intersect.

use geo::Intersects;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
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

use crate::extension::geometries;
use crate::extension::single_geometry;

/// `ST_Intersects` between two native geometry operands, computed by the `geo` crate. 
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoIntersects;

impl GeoIntersects {
    /// A lazy `ScalarFnArray` computing the per-row intersection predicate of `a` and `b`; either
    /// may be constant. 
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
        ScalarFnId::new("vortex.geo.intersects")
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

    fn return_dtype(&self, _: &Self::Options, _: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let a = args.get(0)?;
        let b = args.get(1)?;
        match (a.as_opt::<Constant>(), b.as_opt::<Constant>()) {
            (Some(qa), Some(qb)) => {
                let ga = single_geometry(qa.scalar(), ctx)?;
                let gb = single_geometry(qb.scalar(), ctx)?;
                Ok(ConstantArray::new(
                    Scalar::bool(ga.intersects(&gb), Nullability::NonNullable),
                    a.len(),
                )
                .into_array())
            }
            (Some(query), None) => intersects_constant(&b, query.scalar(), ctx),
            (None, Some(query)) => intersects_constant(&a, query.scalar(), ctx),
            (None, None) => {
                let ag = geometries(&a, ctx)?;
                let bg = geometries(&b, ctx)?;
                vortex_ensure!(
                    ag.len() == bg.len(),
                    "geo intersects: operand length mismatch {} vs {}",
                    ag.len(),
                    bg.len()
                );
                let bits = ag.iter().zip(&bg).map(|(x, y)| x.intersects(y));
                Ok(BoolArray::from_iter(bits).into_array())
            }
        }
    }
}

/// Intersection of each row of `operand` with a constant `query` geometry, decoded once and
/// broadcast. Intersection is symmetric, so this serves a constant on either side.
fn intersects_constant(
    operand: &ArrayRef,
    query: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let query = single_geometry(query, ctx)?;
    let geoms = geometries(operand, ctx)?;
    let bits = geoms.iter().map(|g| g.intersects(&query));
    Ok(BoolArray::from_iter(bits).into_array())
}

#[cfg(test)]
mod tests {
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::Polygon;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_error::VortexResult;

    use super::GeoIntersects;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::wkb_geometry_scalar;

    /// Execute a `GeoIntersects` array and read back its per-row booleans.
    fn intersections(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<bool>> {
        Ok(array
            .execute::<Canonical>(ctx)?
            .into_bool()
            .into_bit_buffer()
            .iter()
            .collect())
    }

    /// `ST_Intersects(point, polygon)` is point-in-polygon: a point inside the polygon intersects
    /// it, a point outside does not.
    #[test]
    fn point_in_polygon() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // A 4x4 square anchored at the origin, as a constant query polygon.
        let square = polygon_column(vec![vec![vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]]])?;
        let square = ConstantArray::new(square.execute_scalar(0, &mut ctx)?, 2).into_array();

        // (2,2) is inside the square; (5,5) is outside.
        let points = point_column(vec![2.0, 5.0], vec![2.0, 5.0])?;
        let result = GeoIntersects::try_new_array(points, square)?.into_array();

        assert_eq!(intersections(result, &mut ctx)?, vec![true, false]);
        Ok(())
    }

    /// The Q1 pushdown path: the polygon arrives as a `WellKnownBinary` constant (a folded geometry
    /// literal), decoded to `geo_types` via the `wkb` crate in `geometries`.
    #[test]
    fn point_in_polygon_wkb_constant() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        // The same 4x4 square, but as a WKB literal rather than a native polygon.
        let square = Geometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));
        let square = ConstantArray::new(wkb_geometry_scalar(&square)?, 2).into_array();

        let points = point_column(vec![2.0, 5.0], vec![2.0, 5.0])?;
        let result = GeoIntersects::try_new_array(points, square)?.into_array();

        assert_eq!(intersections(result, &mut ctx)?, vec![true, false]);
        Ok(())
    }
}
