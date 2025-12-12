// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_vector::Datum;

use crate::Array;
use crate::ArrayRef;
use crate::executor::VectorExecutor;

/// Executor for exporting a Vortex [`Mask`] from an [`ArrayRef`].
pub trait MaskExecutor {
    /// Execute the array to produce a mask.
    fn execute_mask(&self, session: &VortexSession) -> VortexResult<Mask>;
}

impl MaskExecutor for ArrayRef {
    fn execute_mask(&self, session: &VortexSession) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", self.dtype());
        }

        Ok(match self.execute_datum(session)? {
            Datum::Scalar(s) => Mask::new(self.len(), s.as_bool().value().unwrap_or(false)),
            Datum::Vector(v) => {
                let (bits, mask) = v.into_bool().into_parts();
                // To handle nullable boolean arrays, we treat nulls as false in the mask.
                // TODO(ngates): is this correct? Feels like we should just force the caller to
                //  pass non-nullable boolean arrays.
                mask.bitand(&Mask::from(bits))
            }
        })
    }
}
