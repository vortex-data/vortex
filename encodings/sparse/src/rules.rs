// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::CastReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::SparseVTable;

pub(crate) static RULES: ParentRuleSet<SparseVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&CastReduceAdaptor(SparseVTable))]);
