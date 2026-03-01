// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::RLEVTable;

pub(crate) const RULES: ParentRuleSet<RLEVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(RLEVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(RLEVTable)),
]);
