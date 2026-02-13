// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::StructVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::compute::CastExecuteAdaptor;
use crate::expr::ZipExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<StructVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(StructVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(StructVTable)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(StructVTable)),
]);
