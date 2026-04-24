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

use crate::ALP;

pub(super) const PARENT_KERNELS: ParentKernelSet<ALP> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(ALP)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(ALP)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(ALP)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ALP)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ALP)),
]);

pub(super) const RULES: ParentRuleSet<ALP> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BetweenReduceAdaptor(ALP)),
    ParentRuleSet::lift(&CastReduceAdaptor(ALP)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ALP)),
]);
