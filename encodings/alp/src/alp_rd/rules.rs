// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::CastReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::alp_rd::ALPRDVTable;

pub(crate) static RULES: ParentRuleSet<ALPRDVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&CastReduceAdaptor(ALPRDVTable))]);
