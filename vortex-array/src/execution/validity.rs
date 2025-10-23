// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::FutureExt;
use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::execution::{BindCtx, MaskExecution};
use crate::validity::Validity;

impl dyn BindCtx + '_ {
    /// Bind a validity helper into a [`MaskExecution`].
    pub fn bind_validity(
        &mut self,
        validity: &Validity,
        array_len: usize,
        selection: Option<&ArrayRef>,
    ) -> VortexResult<MaskExecution> {
        match selection {
            None => match validity {
                Validity::NonNullable | Validity::AllValid => Ok(MaskExecution::AllTrue(array_len)),
                Validity::AllInvalid => Ok(MaskExecution::AllFalse(array_len)),
                Validity::Array(validity) => self.bind_mask(validity),
            },
            Some(selection) => {
                let selection = self.bind_mask(selection)?;
                match validity {
                    Validity::NonNullable | Validity::AllValid => Ok(MaskExecution::Future(
                        async move { Ok(Mask::AllTrue(selection.execute().await?.true_count())) }
                            .boxed(),
                    )),
                    Validity::AllInvalid => Ok(MaskExecution::Future(
                        async move { Ok(Mask::AllFalse(selection.execute().await?.true_count())) }
                            .boxed(),
                    )),
                    Validity::Array(validity) => {
                        let validity = self.bind_mask(validity)?;
                        Ok(MaskExecution::Future(
                            async move {
                                let validity = validity.execute().await?;
                                let selection = selection.execute().await?;
                                // We perform a take on the validity mask using the selection mask.
                                Ok(validity.filter(&selection))
                            }
                            .boxed(),
                        ))
                    }
                }
            }
        }
    }
}
