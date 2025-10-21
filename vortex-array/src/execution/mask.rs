// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::FutureExt;
use pin_project_lite::pin_project;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::execution::BindCtx;
use crate::ArrayRef;

pin_project! {
    /// A batch execution that produces a Vortex `Mask`.
    #[project = MaskExecutionProj]
    pub enum MaskExecution {
        AllTrue { len: usize },
        AllFalse { len: usize },
        Future { #[pin] fut: BoxFuture<'static, VortexResult<Mask>> },
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

impl dyn BindCtx + '_ {
    /// Bind a non-nullable boolean array into a `MaskExecution`.
    ///
    /// This binding will optimize for constant arrays or other array types that can be more
    /// efficiently converted into a `Mask`.
    fn bind_mask(&mut self, mask: &ArrayRef, ctx: &mut dyn BindCtx) -> VortexResult<MaskExecution> {
        if !matches!(mask.dtype(), DType::Bool(NonNullable)) {
            vortex_bail!(
                "Expected non-nullable boolean array for mask binding, got {}",
                mask.dtype()
            );
        }

        // Check for a constant mask
        if let Some(scalar) = mask.as_constant() {
            let constant = scalar
                .as_bool()
                .value()
                .vortex_expect("checked non-nullable");
            let len = mask.len();
            if constant {
                return Ok(MaskExecution::AllTrue { len });
            } else {
                return Ok(MaskExecution::AllFalse { len });
            }
        }

        // TODO(ngates): we may want to support creating masks from iterator of slices, in which
        //  case we could check for run-end encoding here?

        // If none of the above patterns match, we fall back to canonicalizing.
        let _execution = ctx.bind(mask, None)?;
        // Ok(Self::Future {
        //     fut: async move { Ok(execution.execute().await?.into_bool().to_mask()) }.boxed(),
        // })
        todo!("Finish bind_mask implementation")
    }
}
