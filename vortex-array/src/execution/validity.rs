// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use futures::future::FutureExt;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::execution::{BindCtx, MaskExecution};
use crate::validity::Validity;
use crate::ArrayRef;
use crate::stream::ArrayStreamExt;

/// A batch execution that produces a Vortex `Mask`.
pub struct ValidityExecution(BoxFuture<'static, VortexResult<Mask>>);
impl ValidityExecution {
    pub async fn execute(self) -> VortexResult<Mask> {
        self.0.await
    }
}

impl dyn BindCtx + '_ {
    /// Bind a validity helper into a `ValidityExecution`.
    pub fn bind_validity(
        &mut self,
        validity: &Validity,
        array_len: usize,
        selection: Option<&ArrayRef>,
    ) -> VortexResult<MaskExecution> {
        match selection {
            None => {
                match validity {
                    Validity::NonNullable | Validity::AllValid => {
                        return Ok(MaskExecution::AllTrue(array_len));
                    }
                    Validity::AllInvalid => {
                        return Ok(MaskExecution::AllFalse(array_len));
                    }
                    Validity::Array(validity) => {
                        return self.bind_mask(validity);
                    }
                }
            }
            Some(selection) => {
                let selection_exec = self.bind_mask(selection)?;
                match validity {
                    Validity::NonNullable | Validity::AllValid => {
                        return Ok(MaskExecution::Future(async move {
                            Ok(Mask::AllTrue(selection_exec.execute().await?.true_count()))
                        }.boxed()));
                    }
                    Validity::AllInvalid => {
                        return Ok(MaskExecution::Future(async move {
                            Ok(Mask::AllFalse(selection_exec.execute().await?.true_count()))
                        }.boxed()));
                    }
                    Validity::Array(validity) => {
                        let validity_exec = self.bind_mask(validity)?;
                        return Ok(MaskExecution::Future(async move {
                            let validity = validity_exec.execute().await?;
                            let selection = selection_exec.execute().await?;

                            // We need to perform a take on the validity mask using the selection mask.
                            validity.

                            Ok(selection_mask.filter(&validity_mask))
                        }.boxed()));
                    }
                }
            }
        }
        let Some(selection) = selection {

        } else {
            match validity {
                Validity::NonNullable | Validity::AllValid => {

                    return Ok(ValidityExecution(Box::pin(async move {
                        Ok(Mask::new_true(array_len))
                    })));
                }
                Validity::AllInvalid => {
                    return Ok(ValidityExecution(Box::pin(async move {
                        Ok(Mask::new_false(array_len))
                    })));
                }
            }
        }

        // Check for a constant mask
        if let Some(scalar) = mask.as_constant() {
            let constant = scalar
                .as_bool()
                .value()
                .vortex_expect("checked non-nullable");
            let len = mask.len();
            if constant {
                return Ok(ValidityExecution::AllTrue { len });
            } else {
                return Ok(ValidityExecution::AllFalse { len });
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
