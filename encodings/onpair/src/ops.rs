// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::decode::OwnedDecodeInputs;

impl OperationsVTable<OnPair> for OnPair {
    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let inputs = OwnedDecodeInputs::collect(array, ctx)?;
        let dv = inputs.view();
        let mut buf: Vec<u8> = Vec::with_capacity(dv.decoded_len(index));
        dv.decode_row_into(index, &mut buf);
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
