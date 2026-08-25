// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Length`: planar (Euclidean) length of native lineal geometries.

use geo::Euclidean;
use geo::Length;
use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::unstable::row::RowFn;
use vortex_array::scalar_fn::unstable::row::RowVisitor;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::LineString;
use crate::extension::MultiLineString;
use crate::scalar_fn::row::GeometryRow;

/// Validate the native lineal operand accepted by `ST_Length`.
fn validate_length_operand(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: length requires exactly one lineal operand, got {}",
        dtypes.len()
    );
    vortex_ensure!(
        dtypes[0].as_extension_opt().is_some_and(|extension| {
            extension.is::<LineString>() || extension.is::<MultiLineString>()
        }),
        "spatial: length operand {} is not a native LineString or MultiLineString",
        dtypes[0]
    );
    Ok(())
}

fn euclidean_length(geometry: &Geometry<f64>) -> f64 {
    match geometry {
        Geometry::LineString(line) => Euclidean.length(line),
        Geometry::MultiLineString(lines) => Euclidean.length(lines),
        _ => unreachable!("length dispatch validated a native lineal geometry"),
    }
}

/// Planar (Euclidean) `ST_Length` (no geodesic correction) of native `LineString` and
/// `MultiLineString` geometries.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialLength;

impl SpatialLength {
    /// A lazy `ScalarFnArray` computing the per-row length of a lineal geometry operand.
    pub fn try_new(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialLength, EmptyOptions).erased(),
            vec![array],
        )
    }
}

impl RowFn for SpatialLength {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["geometry"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.st.length");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        validate_length_operand(args)?;
        visitor.visit::<(GeometryRow,), f64>(|(geometry,)| euclidean_length(geometry))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::Columnar;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
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
    use vortex_error::vortex_err;

    use super::SpatialLength;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::point_column;

    fn line_constant(
        line: Vec<(f64, f64)>,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let scalar = linestring_column(vec![line])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    #[test]
    fn measures_each_linestring() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let lines = linestring_column(vec![
            vec![(0.0, 0.0), (3.0, 4.0)],
            vec![(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)],
            vec![(1.0, 2.0)],
            vec![],
        ])?;

        let lengths = SpatialLength::try_new(lines)?.into_array();
        let expected = PrimitiveArray::from_iter([5.0f64, 9.0, 0.0, 0.0]).into_array();

        assert_arrays_eq!(lengths, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn measures_each_multilinestring() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let lines = multilinestring_column(vec![
            vec![
                vec![(0.0, 0.0), (3.0, 4.0)],
                vec![(10.0, 10.0), (10.0, 14.0)],
            ],
            vec![],
            vec![vec![], vec![(1.0, 2.0)], vec![(0.0, 0.0), (0.0, 5.0)]],
        ])?;

        let lengths = SpatialLength::try_new(lines)?.into_array();
        let expected = PrimitiveArray::from_iter([9.0f64, 0.0, 5.0]).into_array();

        assert_arrays_eq!(lengths, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn constant_is_computed_once_and_remains_constant() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let lines = line_constant(vec![(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)], 3, &mut ctx)?;

        let result = SpatialLength::try_new(lines)?
            .into_array()
            .execute::<Columnar>(&mut ctx)?;
        let Columnar::Constant(lengths) = result else {
            return Err(vortex_err!("length of a constant should remain constant"));
        };
        assert_eq!(lengths.len(), 3);
        assert_eq!(f64::try_from(lengths.scalar())?, 9.0);
        Ok(())
    }

    #[test]
    fn null_constant_is_all_null() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let dtype = linestring_column(vec![vec![]])?.dtype().as_nullable();
        let lines = ConstantArray::new(Scalar::null(dtype), 2).into_array();

        let result = SpatialLength::try_new(lines)?
            .into_array()
            .execute::<Columnar>(&mut ctx)?;
        let Columnar::Constant(lengths) = result else {
            return Err(vortex_err!(
                "length of a null constant should remain constant"
            ));
        };
        assert_eq!(lengths.len(), 2);
        assert!(lengths.scalar().is_null());
        Ok(())
    }

    #[test]
    fn propagates_nullable_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let lines = MaskedArray::try_new(
            linestring_column(vec![
                vec![(0.0, 0.0), (3.0, 4.0)],
                vec![(0.0, 0.0), (1.0, 1.0)],
                vec![(0.0, 0.0), (0.0, 4.0)],
            ])?,
            Validity::from_iter([true, false, true]),
        )?
        .into_array();

        let expected = PrimitiveArray::new(
            vec![5.0, 0.0, 4.0],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        let lengths = SpatialLength::try_new(lines)?.into_array();
        assert_arrays_eq!(lengths, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn propagates_all_null_rows() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let lines = MaskedArray::try_new(
            linestring_column(vec![
                vec![(0.0, 0.0), (3.0, 4.0)],
                vec![(0.0, 0.0), (0.0, 4.0)],
            ])?,
            Validity::from_iter([false, false]),
        )?
        .into_array();
        let expected =
            PrimitiveArray::new(vec![0.0, 0.0], Validity::from_iter([false, false])).into_array();

        let lengths = SpatialLength::try_new(lines)?.into_array();

        assert_arrays_eq!(lengths, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn rejects_non_lineal_dtype() -> VortexResult<()> {
        let point = point_column(vec![0.0], vec![0.0])?;
        assert!(
            SpatialLength
                .return_dtype(&EmptyOptions, std::slice::from_ref(point.dtype()))
                .is_err()
        );
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(
            SpatialLength
                .return_dtype(&EmptyOptions, &[primitive])
                .is_err()
        );
        Ok(())
    }
}
