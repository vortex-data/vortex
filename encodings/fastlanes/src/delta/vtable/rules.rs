// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
// Kept (with the registration below) so the Delta cast rule can be re-enabled
// once the in-place widening is made correct; see the TODO below.
#[allow(unused_imports)]
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::delta::vtable::Delta;

pub(crate) static RULES: ParentRuleSet<Delta> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Delta)),
    // TODO(joe): fixme, this is incorrect..
    // ParentRuleSet::lift(&CastReduceAdaptor(Delta)),
]);
