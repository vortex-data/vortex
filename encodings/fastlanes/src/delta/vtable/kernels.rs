// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;

use crate::Delta;

pub(crate) const PARENT_KERNELS: ParentKernelSet<Delta> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(Delta)),
    ParentKernelSet::lift(&BetweenExecuteAdaptor(Delta)),
]);
