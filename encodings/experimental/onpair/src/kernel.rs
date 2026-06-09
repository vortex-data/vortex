// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::like::LikeExecuteAdaptor;

use crate::OnPair;

// TODO: implement ListExecute & TakeExecute for OnPair
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&LikeExecuteAdaptor(OnPair)),
]);
