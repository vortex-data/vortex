// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::VarBinView;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<VarBinView>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(VarBinView)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(VarBinView)),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.slice"),
        &SliceReduceAdaptor(VarBinView),
    ),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<VarBinView> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<VarBinView> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
