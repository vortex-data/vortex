// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;

impl OperationsVTable<OnPair> for OnPair {
    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let column = array.column()?;
        let mut buf: Vec<u8> = Vec::with_capacity(column.max_decompress_capacity().max(64));
        column
            .decompress_row(index, &mut buf)
            .map_err(|e| vortex_err!("OnPair decompress failed: {e}"))?;
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }
}
