// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::VarBin;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<VarBin>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(VarBin)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(VarBin)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(VarBin)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<VarBin> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<VarBin> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
