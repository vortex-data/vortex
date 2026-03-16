// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeList;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<FixedSizeList> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(FixedSizeList)),
    ParentRuleSet::lift(&MaskReduceAdaptor(FixedSizeList)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FixedSizeList)),
]);
