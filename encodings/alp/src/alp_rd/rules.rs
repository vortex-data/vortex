// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::alp_rd::ALPRDVTable;

pub(crate) static RULES: ParentRuleSet<ALPRDVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(ALPRDVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ALPRDVTable)),
]);
