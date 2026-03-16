// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::encodings::sparse::Sparse;
use vortex::encodings::sparse::SparseArray;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct SparseFixture;

impl ArrayFixture for SparseFixture {
    fn name(&self) -> &str {
        "sparse.vortex"
    }

    fn description(&self) -> &str {
        "Mostly-null or mostly-default arrays with sparse non-default values"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let sparse_i64_col = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 50 == 0).then_some(i * 1000)),
        );

        let sparse_str: Vec<Option<&str>> = (0..N)
            .map(|i| (i % 20 == 0).then_some("rare_value"))
            .collect();
        let sparse_str_col = VarBinArray::from(sparse_str);

        let sparse_bool_col = BoolArray::from_iter((0..N).map(|i| (i % 100 == 0).then_some(true)));

        let sparse_f64 = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 100 == 0).then_some(i as f64 * 2.71)),
        );

        let sparse_boundary = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i == 0 || i == (N as i64 - 1) || i % 200 == 0).then(|| i * 7)),
        );

        let arr = StructArray::try_new(
            FieldNames::from([
                "sparse_i64",
                "sparse_str",
                "sparse_bool",
                "sparse_f64",
                "sparse_boundary",
            ]),
            vec![
                SparseArray::encode(&sparse_i64_col.into_array(), None)?,
                SparseArray::encode(&sparse_str_col.into_array(), None)?,
                SparseArray::encode(&sparse_bool_col.into_array(), None)?,
                SparseArray::encode(&sparse_f64.into_array(), None)?,
                SparseArray::encode(&sparse_boundary.into_array(), None)?,
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
