// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::BitPacked;

pub(crate) const RULES: ParentRuleSet<BitPacked> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(BitPacked)),
    ParentRuleSet::lift(&SliceReduceAdaptor(BitPacked)),
]);
