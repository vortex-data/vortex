// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execute checked arithmetic at a sufficient integer width.
//!
//! Non-null operand bounds determine the smallest width that can hold both inputs and the result
//! interval. If that interval exceeds the logical width, execution uses the logical width and lets
//! the checked kernel test the actual row pairs. Independent extrema can overestimate the result,
//! so interval overflow alone must not reject an operation.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::Narrow;
use super::NarrowArraySlotsExt;
use super::encoding::integer_types;
use super::rules::rewrap;
use crate::ArrayRef;
use crate::ArrayView;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::execute_numeric;
use crate::scalar_fn::fns::operators::Operator;

pub(super) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Binary.id(), Narrow, NumericExecute);
}

#[derive(Debug)]
struct NumericExecute;

impl ExecuteParentKernel<Narrow> for NumericExecute {
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Narrow>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let operator = match *parent.options {
            Operator::Add => NumericOperator::Add,
            Operator::Sub => NumericOperator::Sub,
            Operator::Mul => NumericOperator::Mul,
            Operator::Div => NumericOperator::Div,
            _ => return Ok(None),
        };

        // Primitive arithmetic must not reinterpret an extension's storage as an integer buffer.
        let Ok(logical_ptype) = PType::try_from(array.dtype()) else {
            return Ok(None);
        };

        let Some(parent_array) = parent.as_opt::<ScalarFn>() else {
            return Ok(None);
        };

        let (lhs, rhs) = match child_idx {
            0 => (
                array.values().clone(),
                stored_values(parent_array.get_child(1)),
            ),
            1 => (
                stored_values(parent_array.get_child(0)),
                array.values().clone(),
            ),
            _ => return Ok(None),
        };

        let result_dtype = parent.dtype();
        if parent.is_empty() {
            return Ok(Some(Canonical::empty(result_dtype).into_array()));
        }

        let (Some(lhs_bounds), Some(rhs_bounds)) = (bounds(&lhs, ctx)?, bounds(&rhs, ctx)?) else {
            return Ok(Some(
                ConstantArray::new(Scalar::null(result_dtype.clone()), parent.len()).into_array(),
            ));
        };

        let ptype = result_bounds(lhs_bounds, rhs_bounds, operator)
            .and_then(|result| {
                let required = Bounds {
                    min: lhs_bounds.min.min(rhs_bounds.min).min(result.min),
                    max: lhs_bounds.max.max(rhs_bounds.max).max(result.max),
                };
                integer_types(logical_ptype).into_iter().find(|ptype| {
                    ptype.byte_width() <= logical_ptype.byte_width() && required.fits(*ptype)
                })
            })
            .unwrap_or(logical_ptype);

        let lhs = lhs.cast(DType::Primitive(ptype, lhs.dtype().nullability()))?;
        let rhs = rhs.cast(DType::Primitive(ptype, rhs.dtype().nullability()))?;
        let values = execute_numeric(&lhs, &rhs, operator, ctx)?;

        rewrap(array, values)
    }
}

fn stored_values(array: &ArrayRef) -> ArrayRef {
    array
        .as_opt::<Narrow>()
        .map_or_else(|| array.clone(), |array| array.values().clone())
}

/// Integer bounds in a domain large enough for all primitive inputs.
#[derive(Clone, Copy)]
struct Bounds {
    min: i128,
    max: i128,
}

impl Bounds {
    fn fits(self, ptype: PType) -> bool {
        let bits = ptype.bit_width();
        if ptype.is_signed_int() {
            let limit = 1i128 << (bits - 1);
            self.min >= -limit && self.max < limit
        } else {
            self.min >= 0 && self.max < (1i128 << bits)
        }
    }
}

fn bounds(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Bounds>> {
    min_max(array, ctx, NumericalAggregateOpts::default())?
        .map(|bounds| {
            let to_i128 = |scalar: &Scalar| {
                if scalar.dtype().is_signed_int() {
                    i64::try_from(scalar).map(i128::from)
                } else {
                    u64::try_from(scalar).map(i128::from)
                }
            };
            Ok(Bounds {
                min: to_i128(&bounds.min)?,
                max: to_i128(&bounds.max)?,
            })
        })
        .transpose()
}

fn result_bounds(lhs: Bounds, rhs: Bounds, operator: NumericOperator) -> Option<Bounds> {
    match operator {
        NumericOperator::Add => Some(Bounds {
            min: lhs.min.checked_add(rhs.min)?,
            max: lhs.max.checked_add(rhs.max)?,
        }),
        NumericOperator::Sub => Some(Bounds {
            min: lhs.min.checked_sub(rhs.max)?,
            max: lhs.max.checked_sub(rhs.min)?,
        }),
        NumericOperator::Mul => {
            let products = [
                lhs.min.checked_mul(rhs.min)?,
                lhs.min.checked_mul(rhs.max)?,
                lhs.max.checked_mul(rhs.min)?,
                lhs.max.checked_mul(rhs.max)?,
            ];

            Some(Bounds {
                min: *products.iter().min()?,
                max: *products.iter().max()?,
            })
        }
        NumericOperator::Div => {
            // Division is monotone on each side of zero. Include the nearest nonzero divisors
            // when the interval crosses zero. Actual zero divisors are left to the checked kernel.
            let divisors = [rhs.min, rhs.max, -1, 1];
            let mut result: Option<Bounds> = None;

            for divisor in divisors {
                if divisor == 0 || divisor < rhs.min || divisor > rhs.max {
                    continue;
                }

                for dividend in [lhs.min, lhs.max] {
                    let quotient = dividend.checked_div(divisor)?;
                    result = Some(result.map_or(
                        Bounds {
                            min: quotient,
                            max: quotient,
                        },
                        |bounds| Bounds {
                            min: bounds.min.min(quotient),
                            max: bounds.max.max(quotient),
                        },
                    ));
                }
            }

            result
        }
    }
}
