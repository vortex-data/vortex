// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::IntoArray;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::match_each_integer_ptype;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;

impl TakeReduce for Null {
    #[expect(clippy::cast_possible_truncation)]
    fn take(array: ArrayView<'_, Null>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        #[expect(deprecated)]
        let indices = indices.to_primitive();

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |T| {
            for index in indices.as_slice::<T>() {
                if (*index as usize) >= array.len() {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(Some(NullArray::new(indices.len()).into_array()))
    }
}

#[allow(dead_code)]
static KEYED_TAKE_RULES: [ParentRuleEntry<Null>; 1] = [ParentRuleSet::lift_id(
    CachedId::new("vortex.dict"),
    &TakeReduceAdaptor::<Null>(Null),
)];

#[allow(dead_code)]
static KEYED_TAKE_RULES_DENSE: ParentRuleDense<Null> = ParentRuleDense::new();

#[allow(dead_code)]
pub(crate) static TAKE_PARENT_RULES: ParentRuleSet<Null> =
    ParentRuleSet::new_indexed(&KEYED_TAKE_RULES, &KEYED_TAKE_RULES_DENSE, &[]);
