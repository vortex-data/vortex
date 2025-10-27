// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::execution::BindCtx;

pub enum MaskExecution {
    AllTrue(usize),
    AllFalse(usize),
    Future(BoxFuture<'static, VortexResult<Mask>>),
}

impl Future for MaskExecution {
    type Output = VortexResult<Mask>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            MaskExecution::AllTrue(len) => {
                let mask = Mask::new_true(*len);
                Poll::Ready(Ok(mask))
            }
            MaskExecution::AllFalse(len) => {
                let mask = Mask::new_false(*len);
                Poll::Ready(Ok(mask))
            }
            MaskExecution::Future(fut) => fut.poll_unpin(cx),
        }
    }
}

impl dyn BindCtx + '_ {
    /// Bind an optional selection mask into a `MaskExecution`.
    ///
    /// The caller must provide a mask length to handle the case where no mask is provided.
    pub fn bind_selection(
        &mut self,
        mask_len: usize,
        mask: Option<&ArrayRef>,
    ) -> VortexResult<MaskExecution> {
        match mask {
            Some(mask) => {
                assert_eq!(mask.len(), mask_len);
                self.bind_mask(mask)
            }
            None => Ok(MaskExecution::AllTrue(mask_len)),
        }
    }

    /// Bind a non-nullable boolean array into a `MaskExecution`.
    ///
    /// This binding will optimize for constant arrays or other array types that can be more
    /// efficiently converted into a `Mask`.
    pub fn bind_mask(&mut self, mask: &ArrayRef) -> VortexResult<MaskExecution> {
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
                return Ok(MaskExecution::AllTrue(len));
            } else {
                return Ok(MaskExecution::AllFalse(len));
            }
        }

        // TODO(ngates): we may want to support creating masks from iterator of slices, in which
        //  case we could check for run-end encoding here?

        // If none of the above patterns match, we fall back to canonicalizing.
        let execution = self.bind(mask, None)?;
        Ok(MaskExecution::Future(
            async move {
                let mask = execution.await?.into_bool();
                Ok(Mask::from(mask.bits().clone()))
            }
            .boxed(),
        ))
    }
}
