// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ListView;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<ListView> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(ListView)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(ListView)),
]);
