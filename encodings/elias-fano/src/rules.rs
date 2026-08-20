// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;

use crate::EliasFano;

/// Reductions an Elias-Fano array can absorb from its parent without reading a buffer: slicing,
/// which costs one metadata field, and casting, inherited from the generic adaptor.
pub(crate) static RULES: ParentRuleSet<EliasFano> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(EliasFano)),
    ParentRuleSet::lift(&SliceReduceAdaptor(EliasFano)),
]);
