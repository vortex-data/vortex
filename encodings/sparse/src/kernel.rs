// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

use crate::Sparse;

pub(crate) static PARENT_KERNELS: ParentKernelSet<Sparse> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&CompareExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Sparse)),
]);
