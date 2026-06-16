// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Primitive;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Primitive> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(Primitive)),
    ParentKernelSet::lift(&CastExecuteAdaptor(Primitive)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Primitive)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Primitive)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(Primitive)),
]);
