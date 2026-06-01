// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use super::array::FSSTView;

pub(crate) static RULES: ParentRuleSet<FSSTView> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(FSSTView))]);
