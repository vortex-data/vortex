// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared kernels for encodings that retain the per-row uncompressed byte length of
//! each element in a non-nullable integer child array (for example `FSST` and `OnPair`).

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

pub trait UncompressedLengthsVTable: VTable {
    fn uncompressed_lengths(array: ArrayView<'_, Self>) -> ArrayRef;

    fn uncompressed_byte_length(array: ArrayView<'_, Self>) -> VortexResult<ArrayRef> {
        let dtype = DType::Primitive(PType::U64, array.dtype().nullability());
        let lengths = Self::uncompressed_lengths(array).cast(dtype.clone())?;
        Ok(match array.validity()? {
            Validity::NonNullable | Validity::AllValid => lengths,
            Validity::Array(v) => lengths.mask(v)?,
            Validity::AllInvalid => {
                ConstantArray::new(Scalar::null(dtype), lengths.len()).into_array()
            }
        })
    }

    fn compare_to_empty(
        array: ArrayView<'_, Self>,
        operator: CompareOperator,
        rhs_nullability: Nullability,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lengths = Self::uncompressed_lengths(array);
        let buffer = match operator {
            CompareOperator::Gte => BitBuffer::new_set(array.len()),
            CompareOperator::Lt => BitBuffer::new_unset(array.len()),
            _ => lengths
                .binary(
                    ConstantArray::new(Scalar::zero_value(lengths.dtype()), lengths.len())
                        .into_array(),
                    operator.into(),
                )?
                .execute(ctx)?,
        };
        Ok(BoolArray::new(
            buffer,
            array.validity()?.union_nullability(rhs_nullability),
        )
        .into_array())
    }
}
