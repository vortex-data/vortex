// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::Pco;

pub(crate) static RULES: ParentRuleSet<Pco> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Pco)),
    ParentRuleSet::lift(&CastReduceAdaptor(Pco)),
]);
