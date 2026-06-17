// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::byte_length::ByteLengthExecuteAdaptor;

use crate::OnPair;

// TODO: implement ListExecute & TakeExecute for OnPair
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&CompareExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&ByteLengthExecuteAdaptor(OnPair)),
]);
