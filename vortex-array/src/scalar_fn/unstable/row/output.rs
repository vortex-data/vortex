// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row output storage whose first array construction can include batch validity.
//!
//! Built-in owned elements retain their initialized payload here. Arbitrary arrays keep the
//! existing validation boundary before the framework attaches input validity.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use super::batch::finalize_kernel_output;
use super::batch::validate_output_metadata;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

/// Initialized row output awaiting input-derived validity.
///
/// Custom output implementations use [`from_array`](Self::from_array). Batch execution validates
/// that array's dtype, length, and all-valid contract before attaching input validity. Built-in
/// elements retain typed storage, so batch execution constructs their output array only once.
pub struct RowOutput(Storage);

enum Storage {
    Primitive { values: ByteBuffer, ptype: PType },
    Bool(BitBuffer),
    Array(ArrayRef),
}

impl RowOutput {
    /// Retain an arbitrary output array for validation by batch execution.
    pub fn from_array(array: ArrayRef) -> Self {
        Self(Storage::Array(array))
    }

    pub(crate) fn primitive<T: NativePType>(values: Buffer<T>) -> Self {
        Self(Storage::Primitive {
            values: values.into_byte_buffer(),
            ptype: T::PTYPE,
        })
    }

    pub(crate) fn boolean(values: BitBuffer) -> Self {
        Self(Storage::Bool(values))
    }

    pub(crate) fn finish(
        self,
        id: ScalarFnId,
        storage_dtype: &DType,
        expected_len: usize,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match self.0 {
            Storage::Primitive { values, ptype } => {
                let dtype = DType::Primitive(ptype, Nullability::NonNullable);
                validate_output_metadata(
                    id,
                    storage_dtype,
                    expected_len,
                    &dtype,
                    values.len() / ptype.byte_width(),
                )?;

                Ok(PrimitiveArray::from_byte_buffer(values, ptype, validity).into_array())
            }
            Storage::Bool(values) => {
                validate_output_metadata(
                    id,
                    storage_dtype,
                    expected_len,
                    &DType::Bool(Nullability::NonNullable),
                    values.len(),
                )?;

                Ok(BoolArray::new(values, validity).into_array())
            }
            Storage::Array(values) => {
                // A safe custom implementation can return any array. Validate its original
                // validity before masking, including rows that input validity will hide.
                let values = finalize_kernel_output(id, storage_dtype, expected_len, values, ctx)?;

                match validity {
                    Validity::NonNullable => Ok(values),
                    Validity::AllValid => values.cast(storage_dtype.as_nullable()),
                    Validity::Array(valid) => values.mask(valid),
                    Validity::AllInvalid => Ok(ConstantArray::new(
                        Scalar::null(storage_dtype.as_nullable()),
                        expected_len,
                    )
                    .into_array()),
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn into_nonnullable_array(self) -> ArrayRef {
        match self.0 {
            Storage::Primitive { values, ptype } => {
                PrimitiveArray::from_byte_buffer(values, ptype, Validity::NonNullable).into_array()
            }
            Storage::Bool(values) => BoolArray::new(values, Validity::NonNullable).into_array(),
            Storage::Array(values) => values,
        }
    }
}
