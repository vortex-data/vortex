// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use crate::arrays::VarBinViewVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<VarBinViewVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(VarBinViewVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(VarBinViewVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(VarBinViewVTable)),
]);
