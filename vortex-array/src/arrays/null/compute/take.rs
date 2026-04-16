// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::match_each_integer_ptype;
use crate::optimizer::rules::ParentRuleSet;

impl TakeReduce for Null {
    #[expect(clippy::cast_possible_truncation)]
    fn take(array: ArrayView<'_, Null>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
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

impl Null {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}
