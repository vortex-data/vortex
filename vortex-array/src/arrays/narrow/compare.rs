// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare Narrow integers at a sufficient stored width.
//!
//! Two children use their wider stored dtype. Constants outside the stored range yield a known
//! comparison result while preserving the array validity.

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::Narrow;
use super::NarrowArraySlotsExt;
use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::integer::integer_dtype;
use crate::dtype::integer::signed_integer_type;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::binary::execute_compare;
use crate::scalar_fn::fns::operators::CompareOperator;

pub(super) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Binary.id(),
        Narrow,
        CompareExecuteAdaptor(Narrow),
    );
}

impl CompareKernel for Narrow {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            return Ok(None);
        }

        if let Some(rhs) = rhs.as_opt::<Narrow>() {
            if let (Some(lhs_type), Some(rhs_type)) = (
                signed_integer_type(lhs.values().dtype()),
                signed_integer_type(rhs.values().dtype()),
            ) {
                let values_type = lhs_type.max(rhs_type);
                return compare_values(
                    &lhs.values()
                        .cast(integer_dtype(values_type, lhs.dtype().nullability()))?,
                    &rhs.values()
                        .cast(integer_dtype(values_type, rhs.dtype().nullability()))?,
                    operator,
                    ctx,
                )
                .map(Some);
            }
            let ptype = if lhs.values().dtype().as_ptype().byte_width()
                >= rhs.values().dtype().as_ptype().byte_width()
            {
                lhs.values().dtype().as_ptype()
            } else {
                rhs.values().dtype().as_ptype()
            };
            let lhs_dtype = DType::Primitive(ptype, lhs.dtype().nullability());
            let rhs_dtype = DType::Primitive(ptype, rhs.dtype().nullability());

            return compare_values(
                &lhs.values().cast(lhs_dtype)?,
                &rhs.values().cast(rhs_dtype)?,
                operator,
                ctx,
            )
            .map(Some);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let storage_dtype = lhs.values().dtype().with_nullability(nullability);
        if let Ok(value) = constant.cast(&storage_dtype) {
            return compare_values(
                lhs.values(),
                &ConstantArray::new(value, lhs.len()).into_array(),
                operator,
                ctx,
            )
            .map(Some);
        }

        // An integer outside the child's type range is ordered against every stored value.
        let below = if signed_integer_type(lhs.dtype()).is_some() {
            crate::integer::scalar_value(&constant)?.as_i256() < crate::dtype::i256::ZERO
        } else {
            false
        };
        let result = match operator {
            CompareOperator::Eq => false,
            CompareOperator::NotEq => true,
            CompareOperator::Lt | CompareOperator::Lte => !below,
            CompareOperator::Gt | CompareOperator::Gte => below,
        };
        let validity = lhs
            .values()
            .validity()?
            .cast_nullability(nullability, lhs.len(), ctx)?;

        Ok(Some(
            BoolArray::try_new(
                BitBuffer::full_in(result, lhs.len(), ctx.allocator().clone()),
                validity,
            )?
            .into_array(),
        ))
    }
}

fn compare_values(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if lhs.is::<Primitive>() && (rhs.is::<Primitive>() || rhs.is::<Constant>()) {
        execute_compare(lhs, rhs, operator, ctx)
    } else {
        lhs.binary(rhs.clone(), operator.into())
    }
}
