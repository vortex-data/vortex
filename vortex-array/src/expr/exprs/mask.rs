// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;

use crate::ArrayRef;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;

/// An expression that masks an input based on a boolean mask.
///
/// Where the mask is true, the input value is retained; where the mask is false, the output is
/// null. In other words, this performs an intersection of the input's validity with the mask.
pub struct Mask;

impl VTable for Mask {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.mask")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("mask"),
            _ => unreachable!("Invalid child index {} for Mask expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "mask(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[1] == DType::Bool(Nullability::NonNullable),
            "The mask argument to 'mask' must be a non-nullable boolean array, got {}",
            arg_dtypes[1]
        );
        Ok(arg_dtypes[0].as_nullable())
    }

    fn evaluate(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        _scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        todo!()
    }

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let [input, mask]: [Datum; _] = args
            .datums
            .try_into()
            .map_err(|_| vortex_err!("Wrong arg count"))?;
        let mask = mask.into_bool();

        match (input, mask) {
            (Datum::Scalar(input), BoolDatum::Scalar(mask)) => {
                let mut result = input;
                result.mask_validity(mask.value().vortex_expect("mask is non-nullable"));
                Ok(Datum::Scalar(result))
            }
            (Datum::Scalar(input), BoolDatum::Vector(mask)) => {
                let mut result = input.repeat(args.row_count).freeze();
                result.mask_validity(&vortex_mask::Mask::from(mask.into_bits()));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Scalar(mask)) => {
                let mut result = input_array;
                result.mask_validity(&vortex_mask::Mask::new(
                    args.row_count,
                    mask.value().vortex_expect("mask is non-nullable"),
                ));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Vector(mask)) => {
                let mut result = input_array;
                result.mask_validity(&vortex_mask::Mask::from(mask.into_bits()));
                Ok(Datum::Vector(result))
            }
        }
    }
}

/// Creates a mask expression that applies the given boolean mask to the input array.
pub fn mask(array: Expression, mask: Expression) -> Expression {
    Mask.new_expr(EmptyOptions, [array, mask])
}
