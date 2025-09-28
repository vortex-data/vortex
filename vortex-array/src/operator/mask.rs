// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::FutureExt;
use futures::future::BoxFuture;
use pin_project_lite::pin_project;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::arrays::ConstantArray;
use crate::operator::{BatchBindCtx, Operator, OperatorRef};

pin_project! {
    #[project = MaskExecutionProj]
    pub enum MaskExecution {
        AllTrue { len: usize },
        AllFalse { len: usize },
        Future { #[pin] fut: BoxFuture<'static, VortexResult<Mask>> },
    }
}

impl MaskExecution {
    pub fn bind(operator: &OperatorRef, ctx: &mut dyn BatchBindCtx) -> VortexResult<Self> {
        if !matches!(operator.dtype(), DType::Bool(Nullability::NonNullable)) {
            vortex_bail!("Invalid operator dtype for mask {}", operator.dtype());
        }

        // Check for a constant mask
        if let Some(array) = operator.as_any().downcast_ref::<ConstantArray>() {
            let constant = array
                .scalar()
                .as_bool()
                .value()
                .vortex_expect("checked non-nullable");
            let len = array.len();
            if constant {
                return Ok(Self::AllTrue { len });
            } else {
                return Ok(Self::AllFalse { len });
            }
        }

        // TODO(ngates): we may want to support creating masks from iterator of slices, in which
        //  case we could check for run-end encoding here?

        // If none of the above patterns match, we fall back to canonicalizing.
        let execution = ctx.bind_project(operator, None)?;
        Ok(Self::Future {
            fut: async move { Ok(execution.execute().await?.into_bool().to_mask()) }.boxed(),
        })
    }
}

impl Future for MaskExecution {
    type Output = VortexResult<Mask>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.project() {
            MaskExecutionProj::AllTrue { len } => Poll::Ready(Ok(Mask::new_true(*len))),
            MaskExecutionProj::AllFalse { len } => Poll::Ready(Ok(Mask::new_false(*len))),
            MaskExecutionProj::Future { mut fut } => fut.poll_unpin(cx),
        }
    }
}
