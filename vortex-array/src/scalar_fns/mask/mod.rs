// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;

use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::EmptyOptions;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::FunctionId;
use crate::expr::functions::VTable;

/// A function that intersects the validity of an array using another array as a mask.
///
/// Where the `mask` array is true, the corresponding v
pub struct MaskFn;
impl VTable for MaskFn {
    type Options = EmptyOptions;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.mask")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn arg_name(&self, _options: &Self::Options, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ArgName::from("input"),
            1 => ArgName::from("mask"),
            _ => unreachable!("unknown"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_types: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_types[1] == DType::Bool(Nullability::NonNullable),
            "The mask argument to 'mask' must be a non-nullable boolean array, got {}",
            arg_types[1]
        );
        Ok(arg_types[0].as_nullable())
    }

    fn execute(&self, _options: &Self::Options, args: &ExecutionArgs) -> VortexResult<Datum> {
        let input = args.input_datums(0).clone();
        let mask = args.input_datums(1).clone().into_bool();
        match (input, mask) {
            (Datum::Scalar(input), BoolDatum::Scalar(mask)) => {
                let mut result = input;
                result.mask_validity(mask.value().vortex_expect("mask is non-nullable"));
                Ok(Datum::Scalar(result))
            }
            (Datum::Scalar(input), BoolDatum::Vector(mask)) => {
                let mut result = input.repeat(args.row_count()).freeze();
                result.mask_validity(&Mask::from(mask.into_parts().0));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Scalar(mask)) => {
                let mut result = input_array;
                result.mask_validity(&Mask::new(
                    args.row_count(),
                    mask.value().vortex_expect("mask is non-nullable"),
                ));
                Ok(Datum::Vector(result))
            }
            (Datum::Vector(input_array), BoolDatum::Vector(mask)) => {
                let mut result = input_array;
                result.mask_validity(&Mask::from(mask.into_parts().0));
                Ok(Datum::Vector(result))
            }
        }
    }
}
