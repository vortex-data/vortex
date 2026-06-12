// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Bool;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Bool> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(Bool)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Bool)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Bool)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(Bool)),
]);
