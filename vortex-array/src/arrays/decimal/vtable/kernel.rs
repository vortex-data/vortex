// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DecimalVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<DecimalVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&CastExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DecimalVTable)),
]);
