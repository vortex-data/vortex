// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterReduceAdaptor;
use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::CastReduceAdaptor;
use vortex_array::scalar_fn::MaskReduceAdaptor;

use crate::ZigZagVTable;

pub(crate) static RULES: ParentRuleSet<ZigZagVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ZigZagVTable)),
]);
