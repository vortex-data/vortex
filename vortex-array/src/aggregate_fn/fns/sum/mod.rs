// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool_accumulator;
mod float_accumulator;
mod int_accumulator;

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use self::bool_accumulator::BoolSumAccumulator;
use self::float_accumulator::FloatSumAccumulator;
use self::int_accumulator::IntSumAccumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::accumulator::Accumulator;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

/// Options for the Sum aggregate function.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SumOptions {
    /// Whether to use checked arithmetic (default: `true`).
    ///
    /// When `true`, integer overflow produces a null result.
    /// When `false`, integer overflow wraps around.
    ///
    /// Note that i64/u64 inputs can still overflow even with type widening,
    /// since they are already at the widest integer type.
    pub checked: bool,
}

impl Display for SumOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.checked {
            write!(f, "SUM(checked)")
        } else {
            write!(f, "SUM(wrapping)")
        }
    }
}

/// Maps an input PType to the widened output PType for sum.
fn sum_output_ptype(ptype: PType) -> PType {
    match ptype {
        PType::U8 | PType::U16 | PType::U32 | PType::U64 => PType::U64,
        PType::I8 | PType::I16 | PType::I32 | PType::I64 => PType::I64,
        PType::F16 | PType::F32 | PType::F64 => PType::F64,
    }
}

/// Computes the sum of numeric or boolean values.
///
/// For primitive numeric types, the output is widened (unsigned -> u64, signed -> i64,
/// float -> f64). For boolean inputs, `true` counts as 1 and `false` as 0, producing
/// a u64 output.
#[derive(Clone)]
pub struct Sum;

impl AggregateFnVTable for Sum {
    type Options = SumOptions;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.sum")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        match input_dtype {
            DType::Bool(_) => Ok(DType::Primitive(PType::U64, Nullability::Nullable)),
            DType::Primitive(p, _) => Ok(DType::Primitive(
                sum_output_ptype(*p),
                Nullability::Nullable,
            )),
            _ => vortex_bail!("Sum requires numeric or boolean input, got {}", input_dtype),
        }
    }

    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn accumulator(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Box<dyn Accumulator>> {
        let checked = options.checked;
        match input_dtype {
            DType::Bool(_) => Ok(Box::new(BoolSumAccumulator::new(checked))),
            DType::Primitive(p, _) => match sum_output_ptype(*p) {
                PType::U64 => Ok(Box::new(IntSumAccumulator::<u64>::new(checked))),
                PType::I64 => Ok(Box::new(IntSumAccumulator::<i64>::new(checked))),
                PType::F64 => Ok(Box::new(FloatSumAccumulator::new())),
                _ => unreachable!(),
            },
            _ => vortex_bail!("Sum requires numeric or boolean input, got {}", input_dtype),
        }
    }
}

#[cfg(test)]
mod tests;
