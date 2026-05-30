// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::OnPairView;

/// Execute-path parent kernels: ListView-speed `filter` and `take`, both
/// metadata-only over the per-row children.
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPairView> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPairView)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(OnPairView)),
]);
