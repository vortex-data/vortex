// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::BitPackedVTable;

pub(crate) const RULES: ParentRuleSet<BitPackedVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(BitPackedVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(BitPackedVTable)),
]);
