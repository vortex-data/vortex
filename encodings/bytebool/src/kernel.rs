// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::BooleanExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;

use crate::ByteBool;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ByteBool> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BooleanExecuteAdaptor(ByteBool)),
    ParentKernelSet::lift(&CastExecuteAdaptor(ByteBool)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ByteBool)),
]);
