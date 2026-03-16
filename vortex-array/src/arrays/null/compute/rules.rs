// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Null;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Null> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Null)),
    ParentRuleSet::lift(&CastReduceAdaptor(Null)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Null)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Null)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Null)),
]);
