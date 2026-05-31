// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use super::array::FSSTView;
use super::array::FSSTViewArraySlotsExt;

impl OperationsVTable<FSSTView> for FSSTView {
    fn scalar_at(
        array: ArrayView<'_, FSSTView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Preconditions (see `OperationsVTable`): `index` is in bounds and non-null.
        let offset: usize = (&array.codes_offsets().execute_scalar(index, ctx)?).try_into()?;
        let end: usize = (&array.codes_ends().execute_scalar(index, ctx)?).try_into()?;

        let compressed = &array.codes_bytes()[offset..end];
        let decoded = ByteBuffer::from(array.decompressor().decompress(compressed));
        Ok(varbin_scalar(decoded, array.dtype()))
    }
}
