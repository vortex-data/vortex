// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Bool;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Bool> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Bool)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Bool)),
]);
