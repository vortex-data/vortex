// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::functions::ExecutionCtx;
use crate::vectors::VectorIntoArray;
use crate::vtable::CanonicalVTable;
use crate::{Array, Canonical};
use itertools::Itertools;
use vortex_error::VortexExpect;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    fn canonicalize(array: &ScalarFnArray) -> Canonical {
        let child_dtypes: Vec<_> = array.children.iter().map(|c| c.dtype().clone()).collect();
        let child_vectors: Vec<_> = array
            .children()
            .iter()
            .map(|child| child.execute())
            .try_collect()
            // FIXME(ngates): canonicalizing really ought to be fallible
            .vortex_expect(
                "Failed to execute child array during canonicalization of ScalarFnArray",
            );

        let ctx = ExecutionCtx::new(array.len, array.dtype.clone(), child_dtypes, child_vectors);

        let result_vector = array
            .scalar_fn
            .execute(&ctx)
            .vortex_expect("Canonicalize should be fallible");

        result_vector.into_array(&array.dtype).to_canonical()
    }
}
