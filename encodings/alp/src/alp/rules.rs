// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::SliceExecuteAdaptor;
use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::compute::CastReduceAdaptor;
use vortex_array::compute::MaskExecuteAdaptor;
use vortex_array::compute::MaskReduceAdaptor;
use vortex_array::expr::BetweenReduceAdaptor;
use vortex_array::expr::CompareExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::optimizer::rules::ParentRuleSet;

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
