// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::logical::LogicalNot;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::Vector;
use vortex_vector::bool::BoolScalar;

use crate::expr::ChildName;
use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::EmptyOptions;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::FunctionId;
use crate::expr::functions::NullHandling;
use crate::expr::functions::VTable;

pub struct NotFn;
impl VTable for NotFn {
    type Options = EmptyOptions;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.not")
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Fixed(1)
    }

    fn null_handling(&self, _options: &Self::Options) -> NullHandling {
        NullHandling::Propagate
    }

    fn arg_name(&self, _: &Self::Options, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Not expression", arg_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_types[0];
        if !matches!(child_dtype, DType::Bool(_)) {
            vortex_bail!(
                "Not expression expects a boolean child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype.clone())
    }

    fn execute(&self, _: &Self::Options, args: &ExecutionArgs) -> VortexResult<Datum> {
        Ok(match args.input_datums(0) {
            Datum::Scalar(Scalar::Bool(sc)) => {
                Datum::Scalar(BoolScalar::new(sc.value().map(|v| !v)).into())
            }
            Datum::Vector(Vector::Bool(vec)) => Datum::Vector(vec.clone().not().into()),
            _ => unreachable!("Not expects a boolean"),
        })
    }
}
