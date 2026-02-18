// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::compute::CastReduceAdaptor;
use vortex_array::compute::ListContainsElementReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::SequenceVTable;

pub(crate) static RULES: ParentRuleSet<SequenceVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(SequenceVTable)),
    ParentRuleSet::lift(&ListContainsElementReduceAdaptor(SequenceVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(SequenceVTable)),
]);
