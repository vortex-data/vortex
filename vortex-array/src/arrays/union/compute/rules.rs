// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Union;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Union> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&MaskReduceAdaptor(Union)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Union)),
]);

// TODO(connor)[Union]: Register TakeReduce only once nullable indices can introduce outer Union
// nulls while preserving the sparse children's dtypes and row alignment.
