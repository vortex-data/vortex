// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Union;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Union> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&MaskReduceAdaptor(Union)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Union)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Union)),
]);
