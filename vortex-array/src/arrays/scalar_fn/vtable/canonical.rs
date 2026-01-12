// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_vector::Datum;
use vortex_vector::Vector;

use crate::Array;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::expr::ExecutionArgs;
use crate::vectors::VectorIntoArray;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    // TODO(joe): fixme move to execute
    fn canonicalize(array: &ScalarFnArray) -> Canonical {
        let child_dtypes: Vec<_> = array.children.iter().map(|c| c.dtype().clone()).collect();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut child_datums = Vec::with_capacity(array.children.len());
        for child in array.children.iter().cloned() {
            let datum = match child.execute::<CanonicalOutput>(&mut ctx).vortex_expect(
                "Failed to execute child array during canonicalization of ScalarFnArray",
            ) {
                CanonicalOutput::Constant(c) => Datum::Scalar(c.scalar().to_vector_scalar()),
                CanonicalOutput::Array(a) => Datum::Vector(
                    a.into_array()
                        .execute::<Vector>(&mut ctx)
                        .vortex_expect("Failed to convert canonical to vector"),
                ),
            };
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

        result_vector.into_array(&array.dtype)
    }
}
