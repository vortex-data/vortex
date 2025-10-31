// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
                    Validity::NonNullable | Validity::AllValid => {
                        Ok(MaskExecution::lazy(move || {
                            Ok(Mask::AllTrue(selection.execute()?.true_count()))
                        }))
                    }
                    Validity::AllInvalid => Ok(MaskExecution::lazy(move || {
                        Ok(Mask::AllFalse(selection.execute()?.true_count()))
                    })),
                    Validity::Array(validity) => {
                        let validity = self.bind_mask(validity)?;
                        Ok(MaskExecution::lazy(move || {
                            let validity = validity.execute()?;
                            let selection = selection.execute()?;
                            // We perform a take on the validity mask using the selection mask.
                            Ok(validity.filter(&selection))
                        }))
                    }
                }
            }
        }
    }
}
