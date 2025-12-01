// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;

use crate::expr::ChildName;
use crate::expr::ExprId;
use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::EmptyOptions;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::VTable;

pub struct IsNullFn;
impl VTable for IsNullFn {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::new_ref("is_null")
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Fixed(1)
    }

    fn arg_name(&self, _: &Self::Options, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for IsNull expression", arg_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_types: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(NonNullable))
    }

    fn execute(&self, _: &Self::Options, args: &ExecutionArgs) -> VortexResult<Datum> {
        Ok(match args.input_datums(0) {
            Datum::Scalar(sc) => Datum::Scalar(sc.is_invalid().into()),
            Datum::Vector(vec) => Vector::Bool(BoolVector::new(
                vec.validity().to_bit_buffer().not(),
                Mask::AllTrue(vec.len()),
            ))
            .into(),
        })
    }
}
