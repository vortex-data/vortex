// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::compute::CastReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::ByteBoolVTable;

pub(crate) static RULES: ParentRuleSet<ByteBoolVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(ByteBoolVTable)),
    ParentRuleSet::lift(&CastReduceAdaptor(ByteBoolVTable)),
]);
