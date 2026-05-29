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
        let len = inputs.decompressed_row_len(index);
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written = inputs.decompress_row_into(index, buf.spare_capacity_mut());
        debug_assert_eq!(written, len);
        // SAFETY: `decompress_row_into` initialised `written` bytes of the
        // spare capacity reserved above.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
