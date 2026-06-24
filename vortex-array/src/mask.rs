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
use crate::validity::Validity;

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

/// An adapter that coerces null elements of a boolean array to `false` before executing it into a
/// [`Mask`]. Created by [`ArrayRef::null_as_false`].
///
/// Use for filter and pruning predicates over nullable data, where SQL semantics treat `NULL` as
/// not matching.
///
/// Prefer `array.null_as_false().execute(ctx)` over `array.fill_null(false)?.execute::<Mask>(ctx)`:
/// `fill_null` on a lazy `ScalarFn` array (e.g. the result of `apply(<predicate>)`) is currently
/// slow because its `validity()` executes the predicate expression.
pub struct NullAsFalse(ArrayRef);

impl ArrayRef {
    /// Returns an adapter that treats null elements of this boolean array as `false` when executed
    /// into a [`Mask`]. See [`NullAsFalse`].
    pub fn null_as_false(self) -> NullAsFalse {
        NullAsFalse(self)
    }
}

impl NullAsFalse {
    /// Executes the boolean array into a [`Mask`], coercing null elements to `false`.
    ///
    /// Canonicalizes the (possibly lazy) array exactly once and folds validity into the value bits
    /// with a single `AND` that reuses the value buffer when it is uniquely owned.
    pub fn execute(self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        let array = self.0;
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", array.dtype());
        }
        // Non-nullable input needs no coercion; defer to the strict `Mask` execution.
        if !array.dtype().is_nullable() {
            return array.execute::<Mask>(ctx);
        }

        let len = array.len();
        Ok(match array.execute::<Columnar>(ctx)? {
            Columnar::Constant(c) => Mask::new(len, c.scalar().as_bool().value().unwrap_or(false)),
            Columnar::Canonical(c) => {
                let bool = c.into_array().execute::<BoolArray>(ctx)?;
                match bool.as_ref().validity()? {
                    Validity::NonNullable | Validity::AllValid => {
                        Mask::from_buffer(bool.into_bit_buffer())
                    }
                    Validity::AllInvalid => Mask::new_false(len),
                    Validity::Array(v) => {
                        let validity_bits = v.execute::<BoolArray>(ctx)?.into_bit_buffer();
                        Mask::from_buffer(bool.into_bit_buffer() & &validity_bits)
                    }
                }
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
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    fn ctx() -> ExecutionCtx {
        array_session().create_execution_ctx()
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
    fn null_as_false_non_nullable() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mask = array.null_as_false().execute(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn null_as_false_treats_null_as_false() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]).into_array();
        let mask = array.null_as_false().execute(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, false, false]));
        Ok(())
    }

    #[test]
    fn null_as_false_null_constant() -> VortexResult<()> {
        let array =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        let mask = array.null_as_false().execute(&mut ctx())?;
        assert_eq!(mask, Mask::new_false(4));
        Ok(())
    }

    #[test]
    fn null_as_false_matches_fill_null_then_mask() -> VortexResult<()> {
        let array =
            BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]).into_array();
        let via_fill_null = array.fill_null(false)?.execute::<Mask>(&mut ctx())?;
        let via_coerce = array.null_as_false().execute(&mut ctx())?;
        assert_eq!(via_coerce, via_fill_null);
        Ok(())
    }
}
