// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::List;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<List> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(List)),
    ParentRuleSet::lift(&MaskReduceAdaptor(List)),
    ParentRuleSet::lift(&SliceReduceAdaptor(List)),
]);
