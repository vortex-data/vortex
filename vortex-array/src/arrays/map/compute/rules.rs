// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Map;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Map> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Map)),
    ParentRuleSet::lift(&CastReduceAdaptor(Map)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Map)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Map)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Map)),
]);
