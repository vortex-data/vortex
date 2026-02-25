// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeListVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::CastReduceAdaptor;
use crate::scalar_fn::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<FixedSizeListVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(FixedSizeListVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(FixedSizeListVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FixedSizeListVTable)),
]);
