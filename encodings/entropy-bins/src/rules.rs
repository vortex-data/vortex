// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::EntropyBins;

pub(crate) static RULES: ParentRuleSet<EntropyBins> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(EntropyBins)),
    ParentRuleSet::lift(&CastReduceAdaptor(EntropyBins)),
    ParentRuleSet::lift(&MaskReduceAdaptor(EntropyBins)),
]);
