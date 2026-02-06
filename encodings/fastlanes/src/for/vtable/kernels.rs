// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::FoRVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<FoRVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(FoRVTable))]);
