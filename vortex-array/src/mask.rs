// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Executable;
use crate::ExecutionCtx;
use crate::Executor;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::columnar::Columnar;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;

impl Executable for Mask {
    /// Executes a boolean array into a [`Mask`].
    ///
    /// The array must have a non-nullable boolean dtype. Use
    /// [`ArrayRef::null_as_false`] to execute a nullable boolean array, coercing null
    /// elements to `false`.
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

/// An [`Executor`] adapter that coerces null elements of a boolean array to `false` before
/// executing into the target type.
///
/// Created by [`ArrayRef::null_as_false`]. The adapter lives on the stack and moves the
/// underlying [`ArrayRef`], so it does not allocate a wrapper array node.
///
/// Use for filter and pruning predicates over nullable data, where SQL semantics treat
/// `NULL` as not matching:
///
/// ```ignore
/// let mask = array.null_as_false().execute::<Mask>(ctx)?;
/// ```
pub struct NullAsFalse(ArrayRef);

impl ArrayRef {
    /// Returns an [`Executor`] that treats null elements of this boolean array as `false`.
    pub fn null_as_false(self) -> NullAsFalse {
        NullAsFalse(self)
    }
}

impl Executor for NullAsFalse {}

impl Executable<NullAsFalse> for Mask {
    /// Executes a boolean array into a [`Mask`], coercing null elements to `false`.
    ///
    /// The validity is folded into the value bits with a single AND that reuses the value
    /// buffer when it is uniquely owned.
    fn execute(source: NullAsFalse, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array = source.0;
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", array.dtype());
        }
        if !array.dtype().is_nullable() {
            return array.execute(ctx);
        }

        let len = array.len();
        Ok(match array.execute::<Columnar>(ctx)? {
            Columnar::Constant(c) => {
                Mask::new(len, c.scalar().as_bool().value().unwrap_or(false))
            }
            Columnar::Canonical(c) => {
                let bool = c.into_array().execute::<BoolArray>(ctx)?;
                match bool.as_ref().validity()? {
                    Validity::NonNullable | Validity::AllValid => {
                        Mask::from_buffer(bool.into_bit_buffer())
                    }
                    Validity::AllInvalid => Mask::new_false(len),
                    Validity::Array(v) => {
                        let validity_bits = v.execute::<BoolArray>(ctx)?.into_bit_buffer();
                        Mask::from_buffer(bool.into_bit_buffer().bitand_in_place(&validity_bits))
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
    use crate::Executor;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    #[test]
    fn null_as_false_non_nullable() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true]).into_array();
        let mask = array.null_as_false().execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn null_as_false_coerces_nulls() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]).into_array();
        let mask = array.null_as_false().execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::from_iter([true, false, false, false]));
        Ok(())
    }

    #[test]
    fn null_as_false_null_constant() -> VortexResult<()> {
        let array =
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), 4).into_array();
        let mask = array.null_as_false().execute::<Mask>(&mut ctx())?;
        assert_eq!(mask, Mask::new_false(4));
        Ok(())
    }

    #[test]
    fn null_as_false_rejects_non_bool() {
        let array = ConstantArray::new(42i32, 4).into_array();
        assert!(array.null_as_false().execute::<Mask>(&mut ctx()).is_err());
    }

    #[test]
    fn mask_rejects_nullable() {
        let array = BoolArray::from_iter([Some(true), None]).into_array();
        assert!(array.execute::<Mask>(&mut ctx()).is_err());
    }
}
