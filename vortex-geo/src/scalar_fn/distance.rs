// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Planar distance between the paired points of two columns.

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
use vortex_session::VortexSession;

use crate::extension::xy_columns;

/// Planar Euclidean distance between `(ax, ay)` and `(bx, by)`.
fn euclidean_distance(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// Expression computing the planar distance between the paired points of two columns. A constant
/// query point is just a [`ConstantArray`](vortex_array::arrays::ConstantArray) operand.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoDistance;

impl GeoDistance {
    /// A lazy `ScalarFnArray` computing the distance between each row of `a` and `b`.
    pub fn try_new_array(a: ArrayRef, b: ArrayRef, len: usize) -> VortexResult<ScalarFnArray> {
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
        // Bulk path: one tight loop over the flat x/y slices, straight into the output buffer.
        let (ax, ay) = xy_columns(&args.get(0)?, ctx)?;
        let (bx, by) = xy_columns(&args.get(1)?, ctx)?;
        let a = ax.as_slice::<f64>().iter().zip(ay.as_slice::<f64>());
        let b = bx.as_slice::<f64>().iter().zip(by.as_slice::<f64>());
        let distances = a
            .zip(b)
            .map(|((&ax, &ay), (&bx, &by))| euclidean_distance(ax, ay, bx, by));
        Ok(PrimitiveArray::from_iter(distances).into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
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
        let distance = GeoDistance::try_new_array(a, b, 4)?.into_array();

        let got: Vec<f64> = (0..4)
            .map(|idx| f64::try_from(&distance.execute_scalar(idx, &mut ctx)?))
            .collect::<VortexResult<_>>()?;
        assert_eq!(got, vec![0.0, 3.0, 4.0, 5.0]);
        Ok(())
    }
}
