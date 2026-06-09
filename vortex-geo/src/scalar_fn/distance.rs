// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Planar distance from a point column to a constant query point.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::extension::coordinate_from_scalar;
use crate::extension::xy_columns;

/// Planar Euclidean distance between `(ax, ay)` and `(bx, by)`.
fn euclidean_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// Planar distance from each point in a point column to a single constant query point. The first
/// operand is the point column, the second the constant query point; column-to-column distance is
/// not supported.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoDistance;

impl GeoDistance {
    /// A lazy `ScalarFnArray` computing the distance from each row of the point column `a` to the
    /// constant query point `b`. The output length is taken from `a`.
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

    fn return_dtype(&self, _: &Self::Options, _: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::F64, Nullability::NonNullable))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // `a` is the point column; `b` is the constant query point, decoded once and broadcast.
        let points = args.get(0)?;
        let Some(query) = args.get(1)?.as_constant() else {
            vortex_bail!("GeoDistance requires a constant query point as its second operand");
        };
        let query = coordinate_from_scalar(&query)?;
        let (xs, ys) = xy_columns(&points, ctx)?;
        let distances = xs
            .as_slice::<f64>()
            .iter()
            .zip(ys.as_slice::<f64>())
            .map(|(&x, &y)| euclidean_distance(x, y, query.x, query.y));
        Ok(PrimitiveArray::from_iter(distances).into_array())
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
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::GeoDistance;
    use super::euclidean_distance;
    use crate::extension::GeoMetadata;
    use crate::extension::Point;

    /// A `Point` column (CRS `EPSG:4326`) over the given x/y coordinates.
    fn point_column(xs: Vec<f64>, ys: Vec<f64>) -> VortexResult<ArrayRef> {
        let storage = StructArray::from_fields(&[
            ("x", PrimitiveArray::from_iter(xs).into_array()),
            ("y", PrimitiveArray::from_iter(ys).into_array()),
        ])?
        .into_array();
        let metadata = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        };
        let dtype = ExtDType::<Point>::try_new(metadata, storage.dtype().clone())?;
        Ok(ExtensionArray::new(dtype.erased(), storage).into_array())
    }

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

    /// The kernel computes planar Euclidean distance (the 3–4–5 triangle).
    #[test]
    fn euclidean_distance_is_planar() {
        assert_eq!(euclidean_distance(0.0, 0.0, 3.0, 4.0), 5.0);
        assert_eq!(euclidean_distance(1.5, -1.5, 1.5, -1.5), 0.0);
    }

    /// `GeoDistance` returns the per-row distance between two point columns (here the second is a
    /// constant query point).
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

    /// Without a constant query point on either side, column-to-column distance is unsupported and
    /// the kernel errors rather than computing it.
    #[test]
    fn distance_requires_constant_query_point() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let a = point_column(vec![0.0, 1.0], vec![0.0, 1.0])?;
        let b = point_column(vec![3.0, 1.0], vec![4.0, 1.0])?;
        let distance = GeoDistance::try_new_array(a, b)?.into_array();

        assert!(distance.execute::<Canonical>(&mut ctx).is_err());
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
}
