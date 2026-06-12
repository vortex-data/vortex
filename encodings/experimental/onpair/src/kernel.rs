// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::OnPair;

// TODO: implement ListExecute & TakeExecute for OnPair
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair))]);
