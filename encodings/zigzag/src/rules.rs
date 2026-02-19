// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterReduceAdaptor;
use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::expr::CastReduceAdaptor;
use vortex_array::expr::MaskReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::ZigZagVTable;

pub(crate) static RULES: ParentRuleSet<ZigZagVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ZigZagVTable)),
]);
