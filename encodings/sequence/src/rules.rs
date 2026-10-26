// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::builtin::cast::CastReduceAdaptor;
use vortex_functions::list_contains::ListContainsElementReduceAdaptor;

use crate::Sequence;

pub(crate) static RULES: ParentRuleSet<Sequence> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Sequence)),
    ParentRuleSet::lift(&ListContainsElementReduceAdaptor(Sequence)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Sequence)),
]);
