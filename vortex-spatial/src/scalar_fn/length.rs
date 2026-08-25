// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Length`: planar (Euclidean) length of native line strings.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::LineString;
use crate::extension::coordinate::ordinates;
use crate::extension::flatten_row_offsets;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_unary;

/// Validate the native line-string operand accepted by `ST_Length`.
fn validate_length_operands(dtypes: &[DType]) -> VortexResult<()> {
    vortex_ensure!(
        dtypes.len() == 1,
        "spatial: length requires exactly one line string operand, got {}",
        dtypes.len()
    );
    vortex_ensure!(
        dtypes[0]
            .as_extension_opt()
            .is_some_and(|extension| extension.is::<LineString>()),
        "spatial: length operand {} is not a native line string",
        dtypes[0]
    );
    Ok(())
}

/// Compute planar lengths directly over native line-string offsets and coordinate buffers.
fn length_array(
    array: ArrayRef,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let storage = array
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    let (row_offsets, coords) = flatten_row_offsets(storage, ctx)?;
    let xs = ordinates(&coords, "x", ctx)?;
    let ys = ordinates(&coords, "y", ctx)?;
    let lengths = Buffer::from_iter(row_offsets.iter().zip(&row_offsets[1..]).map(
        |(&start, &end)| {
            xs[start..end]
                .windows(2)
                .zip(ys[start..end].windows(2))
                .map(|(x, y)| (x[1] - x[0]).hypot(y[1] - y[0]))
                .fold(0.0, |length, segment| length + segment)
        },
    ));
    Ok(PrimitiveArray::new(lengths, validity).into_array())
}

/// Execute length after shared constant/column and null dispatch.
fn execute_length(
    execution: Execution<1, Validity>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match execution.operands {
        [Operand::Constant(line)] => {
            let validity = Validity::from(execution.nullability);
            let one = length_array(ConstantArray::new(line, 1).into_array(), validity, ctx)?;
            Ok(ConstantArray::new(one.execute_scalar(0, ctx)?, execution.len).into_array())
        }
        [Operand::Column(lines)] => length_array(lines, execution.valid, ctx),
    }
}

/// Planar (Euclidean) `ST_Length` (no geodesic correction) of native line strings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SpatialLength;

impl SpatialLength {
    /// A lazy `ScalarFnArray` computing the per-row length of a line string operand.
    pub fn try_new(array: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(SpatialLength, EmptyOptions).erased(),
            vec![array],
        )
    }
}

impl ScalarFnVTable for SpatialLength {
    type Options = EmptyOptions;

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

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("geometry"),
            _ => unreachable!("length has exactly one child"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        validate_length_operands(dtypes)?;
        Ok(DType::Primitive(PType::F64, dtypes[0].nullability()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = args.get(0)?;
        dispatch_unary(
            &array,
            DType::Primitive(PType::F64, array.dtype().nullability()),
            execute_length,
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

    fn is_infallible(&self, _: &Self::Options) -> bool {
        true
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
    fn rejects_non_linestring_dtype() -> VortexResult<()> {
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
