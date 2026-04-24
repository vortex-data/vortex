// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Decimal;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Decimal> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(Decimal)),
    ParentKernelSet::lift(&CastExecuteAdaptor(Decimal)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Decimal)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Decimal)),
]);
