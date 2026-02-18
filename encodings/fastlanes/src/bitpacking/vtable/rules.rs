// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::CastReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::BitPackedVTable;

pub(crate) const RULES: ParentRuleSet<BitPackedVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&CastReduceAdaptor(BitPackedVTable))]);
