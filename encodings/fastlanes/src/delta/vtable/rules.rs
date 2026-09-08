// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::delta::vtable::Delta;

pub(crate) static RULES: ParentRuleSet<Delta> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Delta)),
    ParentRuleSet::lift(&CastReduceAdaptor(Delta)),
]);
