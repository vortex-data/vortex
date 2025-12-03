// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_vector::Datum;

use crate::Array;
use crate::Canonical;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::SCALAR_FN_SESSION;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::ExecutionArgs;
use crate::vectors::VectorIntoArray;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    fn canonicalize(array: &ScalarFnArray) -> Canonical {
        let child_dtypes: Vec<_> = array.children.iter().map(|c| c.dtype().clone()).collect();
        let child_datums: Vec<_> = array
            .children()
            .iter()
            // TODO(ngates): we could make all execution operate over datums
            .map(|child| child.execute(&SCALAR_FN_SESSION).map(Datum::Vector))
            .try_collect()
            // FIXME(ngates): canonicalizing really ought to be fallible
            .vortex_expect(
                "Failed to execute child array during canonicalization of ScalarFnArray",
            );

        let ctx = ExecutionArgs {
            datums: child_datums,
            dtypes: child_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        };

        let result_vector = array
            .bound
            .execute(ctx)
            .vortex_expect("Canonicalize should be fallible")
            .into_vector()
            .vortex_expect("Canonicalize should return a vector");

        result_vector.into_array(&array.dtype).to_canonical()
    }
}
