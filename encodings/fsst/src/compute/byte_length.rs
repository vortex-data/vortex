// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::scalar_fn::fns::byte_length::ByteLengthKernel;
use vortex_array::scalar_fn::fns::uncompressed_lengths::UncompressedLengthsVTable;
use vortex_error::VortexResult;

use crate::FSST;

impl ByteLengthKernel for FSST {
    fn byte_length(
        array: ArrayView<'_, Self>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Self::uncompressed_byte_length(array).map(Some)
    }
}
