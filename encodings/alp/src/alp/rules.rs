// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelDense;
use vortex_array::kernel::ParentKernelEntry;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::between::BetweenReduceAdaptor;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskExecuteAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::ALP;

static KEYED_PARENT_KERNELS: [ParentKernelEntry<ALP>; 5] = [
    ParentKernelSet::lift_id(CachedId::new("vortex.binary"), &CompareExecuteAdaptor(ALP)),
    ParentKernelSet::lift_id(CachedId::new("vortex.filter"), &FilterExecuteAdaptor(ALP)),
    ParentKernelSet::lift_id(CachedId::new("vortex.mask"), &MaskExecuteAdaptor(ALP)),
    ParentKernelSet::lift_id(CachedId::new("vortex.slice"), &SliceExecuteAdaptor(ALP)),
    ParentKernelSet::lift_id(CachedId::new("vortex.dict"), &TakeExecuteAdaptor(ALP)),
];

static KEYED_PARENT_KERNELS_DENSE: ParentKernelDense<ALP> = ParentKernelDense::new();

pub(super) static PARENT_KERNELS: ParentKernelSet<ALP> =
    ParentKernelSet::new_indexed(&KEYED_PARENT_KERNELS, &KEYED_PARENT_KERNELS_DENSE, &[]);

static KEYED_PARENT_RULES: [ParentRuleEntry<ALP>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.between"), &BetweenReduceAdaptor(ALP)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(ALP)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(ALP)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<ALP> = ParentRuleDense::new();

pub(super) static RULES: ParentRuleSet<ALP> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
