// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::compute::CastReduceAdaptor;
use vortex_array::compute::MaskReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::ByteBoolVTable;

pub(crate) static RULES: ParentRuleSet<ByteBoolVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ByteBoolVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ByteBoolVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ByteBoolVTable)),
]);
