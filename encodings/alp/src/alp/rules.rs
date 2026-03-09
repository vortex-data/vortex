// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::between::BetweenReduceAdaptor;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskExecuteAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::ALPVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<ALPVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ALPVTable)),
]);

pub(super) const RULES: ParentRuleSet<ALPVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BetweenReduceAdaptor(ALPVTable)),
    ParentRuleSet::lift(&CastReduceAdaptor(ALPVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ALPVTable)),
]);
