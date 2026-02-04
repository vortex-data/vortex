// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::alp_rd::ALPRDVTable;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ALPRDVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SliceExecuteAdaptor(ALPRDVTable))]);
