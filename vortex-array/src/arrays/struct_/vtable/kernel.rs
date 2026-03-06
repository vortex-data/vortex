// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::StructVTable;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<StructVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(StructVTable)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(StructVTable)),
]);
