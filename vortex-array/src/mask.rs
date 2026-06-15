// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::columnar::Columnar;
use crate::dtype::DType;
use crate::dtype::Nullability;

impl Executable for Mask {
    /// Executes a boolean array into a [`Mask`].
    ///
    /// The array must have a non-nullable boolean dtype. To execute a nullable boolean array,
    /// coercing null elements to `false`, first call
    /// [`ArrayRef::fill_null(false)`](crate::builtins::ArrayBuiltins::fill_null).
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if !matches!(array.dtype(), DType::Bool(Nullability::NonNullable)) {
            vortex_bail!(
                "Mask array must have boolean(NonNullable) dtype, not {}",
                array.dtype()
            );
        }

        let array_len = array.len();
        Ok(match array.execute(ctx)? {
            Columnar::Constant(s) => {
                Mask::new(array_len, s.scalar().as_bool().value().unwrap_or(false))
            }
            Columnar::Canonical(a) => {
                let bool = a.into_array().execute::<BoolArray>(ctx)?;
                Mask::from(bool.into_bit_buffer())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    #[test]
    fn mask_non_nullable() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mask = array.execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn mask_rejects_nullable() {
        let array = BoolArray::from_iter([Some(true), None]).into_array();
        assert!(array.execute::<Mask>(&mut ctx()).is_err());
    }

    #[test]
    fn fill_null_then_mask_coerces_nulls() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]).into_array();
        let mask = array.fill_null(false)?.execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, false, false]));
        Ok(())
    }

    #[test]
    fn fill_null_then_mask_null_constant() -> VortexResult<()> {
        let array =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        let mask = array.fill_null(false)?.execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::new_false(4));
        Ok(())
    }
}
