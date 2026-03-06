// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;

use crate::FoRVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<FoRVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(FoRVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(FoRVTable)),
]);
