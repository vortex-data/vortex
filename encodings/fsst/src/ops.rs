// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArray;

impl OperationsVTable<FSST> for FSST {
    fn scalar_at(array: &FSSTArray, index: usize, _ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        let compressed = array.codes().scalar_at(index)?;
        let binary_datum = compressed.as_binary().value().vortex_expect("non-null");

        let decoded_buffer = ByteBuffer::from(array.decompressor().decompress(binary_datum));
        Ok(varbin_scalar(decoded_buffer, array.dtype()))
    }
}
