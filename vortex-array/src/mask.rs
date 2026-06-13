// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::columnar::Columnar;
use crate::dtype::DType;

impl Executable for Mask {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", array.dtype());
        }

        if let Some(constant) = array.as_opt::<Constant>() {
            let mask_value = constant.scalar().as_bool().value().unwrap_or(false);
            return Ok(Mask::new(array.len(), mask_value));
        }

        let array_len = array.len();
        Ok(match array.execute(ctx)? {
            Columnar::Constant(s) => {
                Mask::new(array_len, s.scalar().as_bool().value().unwrap_or(false))
            }
            Columnar::Canonical(a) => {
                let bool = a.into_array().execute::<BoolArray>(ctx)?;
                let mask = bool
                    .as_ref()
                    .validity()?
                    .execute_mask(bool.as_ref().len(), ctx)?;
                let bits = bool.into_bit_buffer();
                // Pruning predicates use nullable typed-null stats for "unknown". Treating null
                // bools as false here preserves that contract: unknown cannot prune.
                mask.bitand(&Mask::from(bits))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;

    #[test]
    fn nullable_bool_nulls_are_false_in_mask() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]).into_array();
        let mask = array.execute::<Mask>(&mut LEGACY_SESSION.create_execution_ctx())?;

        assert_eq!(mask, Mask::from_iter([true, false, false, false]));
        Ok(())
    }
}
