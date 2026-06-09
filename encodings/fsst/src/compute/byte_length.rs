// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ValidityVTable;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::byte_length::ByteLengthKernel;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::FSST;
use crate::array::FSSTArrayExt;

impl ByteLengthKernel for FSST {
    fn byte_length(
        array: ArrayView<'_, Self>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullable = array.dtype().nullability();
        let dtype = DType::Primitive(PType::U64, nullable);
        // Uncompressed lengths are non-nullable and may be less than u64 each
        let lengths = array.uncompressed_lengths().cast(dtype.clone())?;
        Ok(Some(match FSST::validity(array)? {
            Validity::NonNullable | Validity::AllValid => lengths,
            Validity::Array(v) => lengths.mask(v)?,
            Validity::AllInvalid => {
                ConstantArray::new(Scalar::null(dtype), lengths.len()).into_array()
            }
        }))
    }
}
