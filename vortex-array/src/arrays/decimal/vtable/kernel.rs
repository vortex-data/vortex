// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DecimalVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::compute::CastExecuteAdaptor;
use crate::expr::FillNullExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<DecimalVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DecimalVTable)),
]);
