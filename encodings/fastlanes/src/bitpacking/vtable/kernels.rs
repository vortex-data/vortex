// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;

use crate::BitPacked;

pub(crate) const PARENT_KERNELS: ParentKernelSet<BitPacked> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(BitPacked)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(BitPacked)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(BitPacked)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BitPacked)),
]);
