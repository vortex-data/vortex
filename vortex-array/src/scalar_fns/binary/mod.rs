// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_compute::arithmetic::Add;
use vortex_compute::arithmetic::Arithmetic;
use vortex_compute::arithmetic::CheckedArithmetic;
use vortex_compute::arithmetic::Div;
use vortex_compute::arithmetic::Mul;
use vortex_compute::arithmetic::Sub;
use vortex_compute::comparison::Compare;
use vortex_compute::comparison::Equal;
use vortex_compute::comparison::GreaterThan;
use vortex_compute::comparison::GreaterThanOrEqual;
use vortex_compute::comparison::LessThan;
use vortex_compute::comparison::LessThanOrEqual;
use vortex_compute::comparison::NotEqual;
use vortex_compute::logical::KleeneAnd;
use vortex_compute::logical::KleeneOr;
use vortex_compute::logical::LogicalOp;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::PrimitiveDatum;

use crate::expr::ChildName;
use crate::expr::Operator;
use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::FunctionId;
use crate::expr::functions::NullHandling;
use crate::expr::functions::VTable;

pub struct BinaryFn;
impl VTable for BinaryFn {
    type Options = Operator;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.binary")
    }

    fn serialize(&self, op: &Operator) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(pb::BinaryOpts { op: (*op).into() }.encode_to_vec()))
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<Operator> {
        let opts = pb::BinaryOpts::decode(bytes)?;
        Operator::try_from(opts.op)
    }

    fn arity(&self, _options: &Operator) -> Arity {
        Arity::Exact(2)
    }

    fn null_handling(&self, options: &Operator) -> NullHandling {
        match options {
            Operator::And | Operator::Or => NullHandling::AbsorbsNull,
            _ => NullHandling::Propagate,
        }
    }

    fn arg_name(&self, _options: &Operator, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("Binary has only two arguments"),
        }
    }

    fn return_dtype(&self, options: &Operator, arg_types: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_types[0];
        let rhs = &arg_types[1];

        if options.is_arithmetic() {
            if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
                return Ok(lhs.with_nullability(lhs.nullability() | rhs.nullability()));
            }
            vortex_bail!(
                "incompatible types for arithmetic operation: {} {}",
                lhs,
                rhs
            );
        }

        Ok(DType::Bool((lhs.is_nullable() || rhs.is_nullable()).into()))
    }

    fn execute(&self, op: &Operator, args: &ExecutionArgs) -> VortexResult<Datum> {
        let lhs: Datum = args.input_datums(0).clone();
        let rhs: Datum = args.input_datums(1).clone();

        match op {
            Operator::Eq => Ok(Compare::<Equal>::compare(lhs, rhs).into()),
            Operator::NotEq => Ok(Compare::<NotEqual>::compare(lhs, rhs).into()),
            Operator::Lt => Ok(Compare::<LessThan>::compare(lhs, rhs).into()),
            Operator::Lte => Ok(Compare::<LessThanOrEqual>::compare(lhs, rhs).into()),
            Operator::Gt => Ok(Compare::<GreaterThan>::compare(lhs, rhs).into()),
            Operator::Gte => Ok(Compare::<GreaterThanOrEqual>::compare(lhs, rhs).into()),
            Operator::And => Ok(<BoolDatum as LogicalOp<KleeneAnd>>::op(
                lhs.into_bool(),
                rhs.into_bool(),
            )
            .into()),
            Operator::Or => {
                Ok(<BoolDatum as LogicalOp<KleeneOr>>::op(lhs.into_bool(), rhs.into_bool()).into())
            }
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                execute_arithmetic_primitive(lhs.into_primitive(), rhs.into_primitive(), *op)
            }
        }
    }
}

fn execute_arithmetic_primitive(
    lhs: PrimitiveDatum,
    rhs: PrimitiveDatum,
    op: Operator,
) -> VortexResult<Datum> {
    // Float arithmetic - no overflow checking needed
    if lhs.ptype().is_float() && lhs.ptype() == rhs.ptype() {
        let result: PrimitiveDatum = match op {
            Operator::Add => Arithmetic::<Add>::eval(lhs, rhs),
            Operator::Sub => Arithmetic::<Sub>::eval(lhs, rhs),
            Operator::Mul => Arithmetic::<Mul>::eval(lhs, rhs),
            Operator::Div => Arithmetic::<Div>::eval(lhs, rhs),
            _ => unreachable!("Not an arithmetic operator"),
        };
        return Ok(result.into());
    }

    // Integer arithmetic - use checked operations
    let result: Option<PrimitiveDatum> = match op {
        Operator::Add => CheckedArithmetic::<Add>::checked_eval(lhs, rhs),
        Operator::Sub => CheckedArithmetic::<Sub>::checked_eval(lhs, rhs),
        Operator::Mul => CheckedArithmetic::<Mul>::checked_eval(lhs, rhs),
        Operator::Div => CheckedArithmetic::<Div>::checked_eval(lhs, rhs),
        _ => unreachable!("Not an arithmetic operator"),
    };
    result
        .map(|d| d.into())
        .ok_or_else(|| vortex_err!("Arithmetic overflow/underflow or type mismatch"))
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::PTypeDowncast;
    use vortex_error::VortexExpect;
    use vortex_mask::Mask;
    use vortex_vector::Datum;
    use vortex_vector::Scalar;
    use vortex_vector::Vector;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PScalar;
    use vortex_vector::primitive::PVector;
    use vortex_vector::primitive::PrimitiveScalar;
    use vortex_vector::primitive::PrimitiveVector;

    use crate::expr::Operator;
    use crate::expr::functions::ExecutionArgs;
    use crate::expr::functions::VTable;
    use crate::scalar_fns::binary::BinaryFn;

    #[test]
    fn test_binary() {
        let exec = ExecutionArgs::new(
            100,
            DType::Bool(NonNullable),
            vec![I32.into(), I32.into()],
            vec![
                Datum::Scalar(Scalar::Primitive(PrimitiveScalar::I32(PScalar::new(Some(
                    2i32,
                ))))),
                Datum::Scalar(Scalar::Primitive(PrimitiveScalar::I32(PScalar::new(Some(
                    3i32,
                ))))),
            ],
        );

        let x = BinaryFn
            .execute(&Operator::Gte, &exec)
            .vortex_expect("shouldnt fail");
        assert!(
            !x.into_scalar()
                .vortex_expect("")
                .into_bool()
                .value()
                .vortex_expect("not null")
        );
        let x = BinaryFn
            .execute(&Operator::Lt, &exec)
            .vortex_expect("shouldnt fail");
        assert!(
            x.into_scalar()
                .vortex_expect("")
                .into_bool()
                .value()
                .vortex_expect("not null")
        );
    }

    #[test]
    fn test_add() {
        let exec = ExecutionArgs::new(
            3,
            DType::Primitive(I32, NonNullable),
            vec![I32.into(), I32.into()],
            vec![
                Datum::Scalar(Scalar::Primitive(PrimitiveScalar::I32(PScalar::new(Some(
                    2i32,
                ))))),
                Datum::Vector(Vector::Primitive(PrimitiveVector::I32(PVector::new(
                    buffer![1, 2, 3],
                    Mask::AllTrue(3),
                )))),
            ],
        );

        let result = BinaryFn
            .execute(&Operator::Add, &exec)
            .vortex_expect("add should succeed");

        let result_vec = result
            .into_vector()
            .vortex_expect("expected vector result")
            .into_primitive();
        let result_i32: PVector<i32> = result_vec.into_i32();
        assert_eq!(result_i32.elements(), &buffer![3, 4, 5]);
        assert_eq!(result_i32.validity(), &Mask::AllTrue(3));
    }
}
