// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::fixed_size_list::FixedSizeListVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<FixedSizeListVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(FixedSizeListVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(FixedSizeListVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FixedSizeListVTable)),
]);
