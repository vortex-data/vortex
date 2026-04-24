// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Dict> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(Dict)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Dict)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Dict)),
]);
