// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;

use crate::SequenceVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<SequenceVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(SequenceVTable)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(SequenceVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(SequenceVTable)),
]);
