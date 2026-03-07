// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::ByteBoolVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ByteBoolVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(ByteBoolVTable))]);
