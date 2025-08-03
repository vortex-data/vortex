// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::{BindContext, Encoding, Evaluation, EvaluationContext};
use crate::experiment::mask::BitMask;
use crate::experiment::vector::{N, Vector};
use std::task::{Poll, ready};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_expr::Operator;
use vortex_scalar::Scalar;

pub struct CompareEncoding {
    lhs: Box<dyn Encoding>,
    rhs: Scalar,
    operator: Operator,
}

impl CompareEncoding {
    pub fn new(lhs: Box<dyn Encoding>, operator: Operator, rhs: Scalar) -> Self {
        Self { lhs, rhs, operator }
    }
}

impl Encoding for CompareEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let lhs = self.lhs.bind(ctx)?;
        let ptype = ctx.dtype.as_ptype();

        match_each_native_ptype!(ptype, |T| {
            let op = match self.operator {
                Operator::Eq => |lhs: &T, rhs: &T| lhs == rhs,
                Operator::NotEq => |lhs: &T, rhs: &T| lhs != rhs,
                Operator::Lt => |lhs: &T, rhs: &T| lhs < rhs,
                Operator::Lte => |lhs: &T, rhs: &T| lhs <= rhs,
                Operator::Gt => |lhs: &T, rhs: &T| lhs > rhs,
                Operator::Gte => |lhs: &T, rhs: &T| lhs >= rhs,
                _ => vortex_bail!("Unsupported operator for comparison: {}", self.operator),
            };

            let rhs = self
                .rhs
                .as_primitive()
                .typed_value::<T>()
                .ok_or_else(|| vortex_err!("Does not support comparison with null"))?;

            Ok(Box::new(ComparePrimitive {
                lhs,
                rhs,
                op,
                lhs_elems: [T::default(); N],
            }) as Box<dyn Evaluation>)
        })
    }
}

struct ComparePrimitive<T, F> {
    lhs: Box<dyn Evaluation>,
    rhs: T,
    op: F,

    // Reusable buffer of elements for the child to export into.
    lhs_elems: [T; N],
}

impl<T: NativePType, F> Evaluation for ComparePrimitive<T, F>
where
    F: Fn(&T, &T) -> bool,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.lhs.seek(chunk_idx)
    }

    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &BitMask,
        defined: &BitMask,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>> {
        let mut elems = Vector::new_primitive::<T>(self.lhs_elems.as_mut(), None);
        ready!(self.lhs.step(ctx, selected, defined, &mut elems))?;

        // FIXME(ngates): we need to look at the selected and defined masks to determine

        // Now we compare each element in `elems` with `self.rhs`.
        let mut out_bool = out.as_bool();
        let out_bits = out_bool.as_mut();
        for (i, item) in self.lhs_elems.iter().enumerate() {
            out_bits.set(i, (self.op)(item, &self.rhs));
        }

        Poll::Ready(Ok(()))
    }
}
