// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::take_referenced_canonical;
use vortex_error::VortexResult;

use crate::OnPair;

impl TakeExecute for OnPair {
    /// Decodes each referenced row once and gathers the decoded strings.
    ///
    /// Gathering token runs instead would decode a row again for every index that repeats it,
    /// which is far slower when the indices are dense, as in a dictionary over OnPair values.
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_referenced_canonical(array.array(), indices, ctx).map(Some)
    }
}
