// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ListVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<ListVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ListVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ListVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ListVTable)),
]);
