// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::Array;
use crate::Canonical;
use crate::LEGACY_SESSION;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::VectorExecutor;
use crate::expr::ExecutionArgs;
use crate::vectors::VectorIntoArray;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    fn canonicalize(array: &ScalarFnArray) -> Canonical {
        let child_dtypes: Vec<_> = array.children.iter().map(|c| c.dtype().clone()).collect();

        let mut child_datums = Vec::with_capacity(array.children.len());
        for child in array.children.iter() {
            let datum = child.execute_datum(&LEGACY_SESSION).vortex_expect(
                "Failed to execute child array during canonicalization of ScalarFnArray",
            );
            child_datums.push(datum);
        }

        let ctx = ExecutionArgs {
            datums: child_datums,
            dtypes: child_dtypes,
            row_count: array.len,
            return_dtype: array.dtype.clone(),
        };

        let len = array.len;
        let result_vector = array
            .scalar_fn
            .execute(ctx)
            .vortex_expect("Canonicalize should be fallible")
            .unwrap_into_vector(len);

        result_vector.into_array(&array.dtype).to_canonical()
    }
}
