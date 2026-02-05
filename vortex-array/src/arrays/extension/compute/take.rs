// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::TakeReduce;
use crate::arrays::TakeReduceAdaptor;
use crate::compute::{self};
use crate::optimizer::rules::ParentRuleSet;

fn take_extension(array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
    let taken_storage = compute::take(array.storage(), indices)?;
    Ok(ExtensionArray::new(
        array
            .ext_dtype()
            .with_nullability(taken_storage.dtype().nullability()),
        taken_storage,
    )
    .into_array())
}

impl TakeReduce for ExtensionVTable {
    fn take(array: &ExtensionArray, indices: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        take_extension(array, indices).map(Some)
    }
}

impl ExtensionVTable {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}
