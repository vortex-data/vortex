// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::Nullability;
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
        if !matches!(self.dtype(), DType::Bool(Nullability::NonNullable)) {
            vortex_bail!("Mask array must have non-nullable boolean dtype");
        }

        if let Some(constant) = self.as_opt::<ConstantVTable>() {
            let value = constant
                .scalar()
                .as_bool()
                .value()
                .vortex_expect("non-nullable");
            Ok(Mask::new(self.len(), value))
        } else {
            Ok(Mask::from(self.execute(session)?.into_bool().into_bits()))
        }
    }
}
