// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::ALPVTable;

pub(super) const PARENT_RULES: ParentRuleSet<ALPVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterReduceAdaptor(ALPVTable))]);
