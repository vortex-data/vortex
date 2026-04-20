// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Null;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<Null>; 5] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &FilterReduceAdaptor(Null)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Null)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(Null)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Null)),
    ParentRuleSet::lift_id(CachedId::new("vortex.dict"), &TakeReduceAdaptor(Null)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Null> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<Null> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
