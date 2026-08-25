// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::BlockedFoR;

// NOTE: unlike `FoR`, a `Filter` cannot be pushed down through this encoding: filtering
// renumbers rows, which breaks the mapping from a row to the block whose reference it was
// encoded against.
pub(super) const PARENT_RULES: ParentRuleSet<BlockedFoR> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(BlockedFoR))]);
