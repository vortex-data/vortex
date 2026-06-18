// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Distance`: planar (Euclidean) distance between two native geometries via the `geo` crate.

use geo::Distance;
use geo::Euclidean;
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
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::extension::geometries;
use crate::extension::single_geometry;

/// Planar (Euclidean) `ST_Distance` (no geodesic correction) between two native geometry operands.
/// Each is a column or a constant literal; `geo` computes the distance between each pair.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoDistance;

impl GeoDistance {
    /// A lazy `ScalarFnArray` computing the per-row distance between operands `a` and `b`; either may
    /// be constant. The output length is taken from `a`.
    pub fn try_new_array(a: ArrayRef, b: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoDistance, EmptyOptions).erased(),
            vec![a, b],
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

    fn return_dtype(&self, _: &Self::Options, _: &[DType]) -> VortexResult<DType> {
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
        match (a.as_opt::<Constant>(), b.as_opt::<Constant>()) {
            (Some(qa), Some(qb)) => {
                let ga = single_geometry(qa.scalar(), ctx)?;
                let gb = single_geometry(qb.scalar(), ctx)?;
                let distance = Euclidean.distance(&ga, &gb);
                Ok(ConstantArray::new(
                    Scalar::primitive(distance, Nullability::NonNullable),
                    a.len(),
                )
                .into_array())
            }
            (Some(query), None) => distances_to_constant(&b, query.scalar(), ctx),
            (None, Some(query)) => distances_to_constant(&a, query.scalar(), ctx),
            (None, None) => {
                let ag = geometries(&a, ctx)?;
                let bg = geometries(&b, ctx)?;
                vortex_ensure!(
                    ag.len() == bg.len(),
                    "geo distance: operand length mismatch {} vs {}",
                    ag.len(),
                    bg.len()
                );
                let distances = ag.iter().zip(&bg).map(|(x, y)| Euclidean.distance(x, y));
                Ok(PrimitiveArray::from_iter(distances).into_array())
            }
        }
    }
}

/// Distance from each row of `operand` to a constant `query` geometry, decoded once and broadcast.
/// Distance is symmetric, so this serves a constant on either side.
fn distances_to_constant(
    operand: &ArrayRef,
    query: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let query = single_geometry(query, ctx)?;
    let geoms = geometries(operand, ctx)?;
    let distances = geoms.iter().map(|g| Euclidean.distance(g, &query));
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
    use vortex_error::VortexResult;

    use super::GeoDistance;
    use crate::test_harness::point_column;

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

    /// Execute a `GeoDistance` array and read back its per-row `f64` distances.
    fn distances(distance: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<f64>> {
        Ok(distance
            .execute::<Canonical>(ctx)?
            .into_primitive()
            .as_slice::<f64>()
            .to_vec())
    }

    /// `GeoDistance` returns the per-row distance between a point column and a constant query point
    /// (3–4–5 triangles), computed via the geo crate.
    #[test]
    fn distance_over_points() -> VortexResult<()> {
        let session = vortex_array::array_session();
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
        let session = vortex_array::array_session();
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
        let session = vortex_array::array_session();
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
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let a = point_constant(0.0, 0.0, 3, &mut ctx)?;
        let b = point_constant(3.0, 4.0, 3, &mut ctx)?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert_eq!(distances(distance, &mut ctx)?, vec![5.0, 5.0, 5.0]);
        Ok(())
    }
}
