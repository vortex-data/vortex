// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::ByteBool;

pub(crate) static RULES: ParentRuleSet<ByteBool> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ByteBool)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ByteBool)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ByteBool)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ByteBool)),
]);
