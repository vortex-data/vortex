// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operand decoding for the fused primitive comparison path.

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
    /// A varying primitive column and its validity.
    Array {
        /// The materialized values.
        values: Buffer<T>,

        /// The validity of the values.
        validity: Validity,
    },

    /// A non-null value repeated for every row.
    Constant {
        /// The repeated value.
        value: T,

        /// The number of repeated rows.
        len: usize,

        /// The validity implied by the constant's dtype.
        validity: Validity,
    },

    /// An all-null constant with this row count.
    Null(usize),
}

impl<T: NativePType> PrimitiveOperand<T> {
    /// Decode an operand once for the fused comparison loop.
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

    /// Return the logical row count.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    /// Return the operand validity.
    pub(super) fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } => validity.clone(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}
