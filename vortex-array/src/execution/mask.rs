// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::execution::BindCtx;
use crate::ArrayRef;

pub enum MaskExecution {
    AllTrue(usize),
    AllFalse(usize),
    Future(BoxFuture<'static, VortexResult<Mask>>),
}

impl MaskExecution {
    pub async fn execute(self) -> VortexResult<Mask> {
        match self {
            MaskExecution::AllTrue(len) => Ok(Mask::new_true(len)),
            MaskExecution::AllFalse(len) => Ok(Mask::new_false(len)),
            MaskExecution::Future(fut) => fut.await,
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
        let _execution = self.bind(mask, None)?;
        // Ok(Self::Future {
        //     fut: async move { Ok(execution.execute().await?.into_bool().to_mask()) }.boxed(),
        // })
        todo!("Finish bind_mask implementation")
    }
}
