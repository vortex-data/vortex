// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding shared by primitive binary operators.

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::validity::Validity;

/// A materialized primitive column, a non-null constant, or an all-null constant.
pub(super) enum PrimitiveOperand<T: NativePType> {
    Array {
        values: Buffer<T>,
        validity: Validity,
    },
    Constant {
        value: T,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl<T: NativePType> PrimitiveOperand<T> {
    pub(super) fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            return Ok(
                match constant.scalar().as_primitive().try_typed_value::<T>()? {
                    Some(value) => Self::Constant {
                        value,
                        len: array.len(),
                        validity: if constant.scalar().dtype().is_nullable() {
                            Validity::AllValid
                        } else {
                            Validity::NonNullable
                        },
                    },
                    None => Self::Null(array.len()),
                },
            );
        }

        let array = array.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?;
        let values = array.into_buffer::<T>();
        Ok(Self::Array { values, validity })
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    pub(super) fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } => validity.clone(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}
