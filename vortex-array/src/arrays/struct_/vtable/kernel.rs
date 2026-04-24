// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Struct;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Struct> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(Struct)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(Struct)),
]);
