// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::RLE;

pub(crate) const RULES: ParentRuleSet<RLE> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&CastReduceAdaptor(RLE))]);
