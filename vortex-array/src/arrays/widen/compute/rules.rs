// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Widen;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Widen> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Widen)),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Widen)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Widen)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Widen)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Widen)),
]);
