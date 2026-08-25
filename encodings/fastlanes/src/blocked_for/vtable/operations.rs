// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::BlockedFoR;
use crate::blocked_for::array::BLOCK_SIZE;
use crate::blocked_for::array::BlockedFoRArrayExt;
use crate::blocked_for::array::BlockedFoRArraySlotsExt;

impl OperationsVTable<BlockedFoR> for BlockedFoR {
    fn scalar_at(
        array: ArrayView<'_, BlockedFoR>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let dtype = array.as_ref().dtype().clone();
        let block = (index + array.offset() as usize) / BLOCK_SIZE;
        let reference = array.references().execute_scalar(block, ctx)?;
        let reference = reference.as_primitive();
        let encoded = array.encoded().execute_scalar(index, ctx)?;
        let encoded = encoded.as_primitive();

        Ok(match_each_integer_ptype!(dtype.as_ptype(), |P| {
            encoded
                .typed_value::<P>()
                .map(|v| {
                    v.wrapping_add(
                        reference
                            .typed_value::<P>()
                            .vortex_expect("BlockedFoR reference value cannot be null"),
                    )
                })
                .map(|v| Scalar::primitive::<P>(v, dtype.nullability()))
                .unwrap_or_else(|| Scalar::null(dtype.clone()))
        }))
    }
}
