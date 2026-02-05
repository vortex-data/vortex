// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::ALPRDVTable;

pub(super) const PARENT_RULES: ParentRuleSet<ALPRDVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterReduceAdaptor(ALPRDVTable))]);
