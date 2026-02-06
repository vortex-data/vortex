// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DecimalVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<DecimalVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(DecimalVTable))]);
