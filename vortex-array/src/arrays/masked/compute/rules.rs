// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Masked;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<Masked>; 4] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &FilterReduceAdaptor(Masked)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(Masked)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Masked)),
    ParentRuleSet::lift_id(CachedId::new("vortex.dict"), &TakeReduceAdaptor(Masked)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Masked> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<Masked> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
