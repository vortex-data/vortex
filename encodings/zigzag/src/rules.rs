// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::ZigZagVTable;

pub(crate) static RULES: ParentRuleSet<ZigZagVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ZigZagVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ZigZagVTable)),
]);
