// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_dtype::DType;
use vortex_error::vortex_bail;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::arrays::ConstantVTable;
use crate::Array;

impl dyn Array + '_ {
    /// Execute the array to produce a mask.
    pub fn execute_mask(&self, session: &VortexSession) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", self.dtype());
        }

        if let Some(constant) = self.as_opt::<ConstantVTable>() {
            let value = constant
                .scalar()
                .as_bool()
                .value()
                .vortex_expect("non-nullable");
            Ok(Mask::new(self.len(), value))
        } else {
            let bool = self.execute(session)?.into_bool();

            // To handle nullable boolean arrays, we treat nulls as false in the mask.
            // TODO(ngates): is this correct? Feels like we should just force the caller to
            //  pass non-nullable boolean arrays.
            let (bits, mask) = bool.into_parts();
            let mask = mask.bitand(&Mask::from(bits));

            Ok(mask)
        }
    }
}
