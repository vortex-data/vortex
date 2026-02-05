// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::PcoVTable;

pub(crate) static RULES: ParentRuleSet<PcoVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(PcoVTable))]);
