// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::List;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;

pub(crate) const PARENT_KERNELS: ParentKernelSet<List> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(List)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(List)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(List)),
]);
