// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use super::array::FSSTView;

pub(super) const PARENT_KERNELS: ParentKernelSet<FSSTView> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(FSSTView)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(FSSTView)),
]);
