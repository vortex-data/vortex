// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::Affine;

pub(super) const PARENT_RULES: ParentRuleSet<Affine> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(Affine))]);
