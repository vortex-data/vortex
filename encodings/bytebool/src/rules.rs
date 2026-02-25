// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::ByteBoolVTable;

pub(crate) static RULES: ParentRuleSet<ByteBoolVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ByteBoolVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ByteBoolVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ByteBoolVTable)),
]);
