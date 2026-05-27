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
    /// Executes a boolean array into a [`Mask`].
    ///
    /// The array must have a non-nullable boolean dtype. Use [`MaskNullAsFalse`] to execute a
    /// nullable boolean array, coercing null elements to `false`.
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        execute_mask(array, ctx, NullHandling::Reject)
    }
}

/// An [`Executable`] target that executes a boolean array into a [`Mask`], coercing null
/// elements to `false`.
///
/// [`Mask`] itself requires a non-nullable boolean array and errors on nullable input. Use this
/// wrapper for filter and pruning predicates over nullable data, where SQL semantics treat
/// `NULL` as not matching.
pub struct MaskNullAsFalse(Mask);

impl MaskNullAsFalse {
    /// Consumes the wrapper and returns the underlying [`Mask`].
    pub fn into_mask(self) -> Mask {
        self.0
    }
}

impl From<MaskNullAsFalse> for Mask {
    fn from(value: MaskNullAsFalse) -> Self {
        value.0
    }
}

impl Executable for MaskNullAsFalse {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        execute_mask(array, ctx, NullHandling::AsFalse).map(Self)
    }
}

/// How [`execute_mask`] treats null elements of a nullable boolean array.
enum NullHandling {
    /// Error if the boolean array is nullable.
    Reject,
    /// Treat null elements as `false`.
    AsFalse,
}

fn execute_mask(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
    null_handling: NullHandling,
) -> VortexResult<Mask> {
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
            match null_handling {
                NullHandling::Reject => {
                    if bool.as_ref().dtype().is_nullable() {
                        vortex_bail!(
                            "Mask requires a non-nullable boolean array, not {}; \
                             use MaskNullAsFalse to coerce nulls to false",
                            bool.as_ref().dtype()
                        );
                    }
                    Mask::from(bool.into_bit_buffer())
                }
                NullHandling::AsFalse => {
                    let validity = bool
                        .as_ref()
                        .validity()?
                        .execute_mask(bool.as_ref().len(), ctx)?;
                    validity.bitand(&Mask::from(bool.into_bit_buffer()))
                }
            }
        }
    })
}
