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

/// Executes a boolean array into a [`Mask`], coercing null elements to `false` (SQL-style
/// `NULL`-as-not-matching semantics).
///
/// This is the cheap counterpart to [`ArrayRef::fill_null(false)`](crate::builtins::ArrayBuiltins::fill_null)
/// followed by [`Mask::execute`]. It canonicalizes the (possibly lazy) array exactly once and folds
/// validity into the value bits with a single bitmap `AND`, rather than routing through the generic
/// `fill_null` scalar function, which derives validity from the lazy expression tree (re-evaluating
/// predicates such as `LIKE`) and materializes an intermediate array.
pub fn execute_mask_coercing_nulls(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    if !matches!(array.dtype(), DType::Bool(_)) {
        vortex_bail!("Mask array must have boolean dtype, not {}", array.dtype());
    }

    let array_len = array.len();
    Ok(match array.execute(ctx)? {
        Columnar::Constant(s) => {
            Mask::new(array_len, s.scalar().as_bool().value().unwrap_or(false))
        }
        Columnar::Canonical(a) => {
            let bool = a.into_array().execute::<BoolArray>(ctx)?;
            // Treat nulls as `false`: fold validity into the value bits.
            let validity = bool
                .as_ref()
                .validity()?
                .execute_mask(bool.as_ref().len(), ctx)?;
            let bits = bool.into_bit_buffer();
            validity.bitand(&Mask::from(bits))
        }
    })
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
    use crate::mask::execute_mask_coercing_nulls;
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

    #[test]
    fn coercing_nulls_non_nullable() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mask = execute_mask_coercing_nulls(array, &mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn coercing_nulls_treats_null_as_false() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]).into_array();
        let mask = execute_mask_coercing_nulls(array, &mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, false, false]));
        Ok(())
    }

    #[test]
    fn coercing_nulls_null_constant() -> VortexResult<()> {
        let array =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        let mask = execute_mask_coercing_nulls(array, &mut ctx())?;
        assert_eq!(mask, Mask::new_false(4));
        Ok(())
    }

    #[test]
    fn coercing_nulls_matches_fill_null_then_mask() -> VortexResult<()> {
        let array =
            BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]).into_array();
        let via_fill_null = array
            .clone()
            .fill_null(false)?
            .execute::<Mask>(&mut ctx())?;
        let via_coerce = execute_mask_coercing_nulls(array, &mut ctx())?;
        assert_eq!(via_coerce, via_fill_null);
        Ok(())
    }
}
