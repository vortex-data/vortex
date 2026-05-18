// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex_array::scalar_fn::fns::like::LikeExecuteAdaptor;

use crate::OnPair;

// TODO: implement TakeExecute for OnPair to add a TakeExecuteAdaptor here
// (matches the FSST pattern; would dispatch take on the codes child + reuse
// the dictionary, mirroring the slice path).
pub(super) const PARENT_KERNELS: ParentKernelSet<OnPair> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CastExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&CompareExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(OnPair)),
    ParentKernelSet::lift(&LikeExecuteAdaptor(OnPair)),
]);
