// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::OnPairView;

/// Reduce-path parent rules: metadata-only `slice`, `take` and `cast`.
pub(crate) static RULES: ParentRuleSet<OnPairView> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(OnPairView)),
    ParentRuleSet::lift(&TakeReduceAdaptor(OnPairView)),
    ParentRuleSet::lift(&CastReduceAdaptor(OnPairView)),
]);
