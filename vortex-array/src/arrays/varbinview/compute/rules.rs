// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use crate::arrays::SliceReduceAdaptor;
use crate::arrays::VarBinViewVTable;
use crate::compute::CastReduceAdaptor;
use crate::expr::MaskReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBinViewVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBinViewVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(VarBinViewVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBinViewVTable)),
]);
