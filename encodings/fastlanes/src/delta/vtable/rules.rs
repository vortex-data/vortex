// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::delta::vtable::Delta;

pub(crate) static RULES: ParentRuleSet<Delta> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Delta)),
    // TODO(joe): fixme, this is incorrect..
    // ParentRuleSet::lift(&CastReduceAdaptor(Delta)),
]);
