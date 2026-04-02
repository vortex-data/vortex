// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::Nullability;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::encodings::sparse::Sparse;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct SparseFixture;

impl FlatLayoutFixture for SparseFixture {
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
        let sparse_str_col = VarBinArray::from_nullable_strs(sparse_str);

        let sparse_bool_col = BoolArray::from_iter((0..N).map(|i| (i % 100 == 0).then_some(true)));

        let sparse_f64 = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 100 == 0).then_some(i as f64 * 2.71)),
        );

        let sparse_boundary = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i == 0 || i == (N as i64 - 1) || i % 200 == 0).then(|| i * 7)),
        );
        let explicit_fill_values = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| if i % 75 == 0 { Some(99) } else { Some(10) }),
        );
        let all_default = ConstantArray::new(10i32, N).into_array();
        let clustered_edges = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i < 8 || i >= N as i64 - 8).then(|| i * 9)),
        );
        let almost_dense = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| if i % 32 == 0 { None } else { Some((i % 5) + 1) }),
        );
        let mixed_null_and_values = PrimitiveArray::from_option_iter((0..N as i32).map(|i| {
            if i % 17 == 0 {
                None
            } else if i % 19 == 0 {
                Some(77)
            } else {
                Some(0)
            }
        }));
        let mixed_null_fill = Scalar::null(mixed_null_and_values.dtype().clone());

        let arr = StructArray::try_new(
            FieldNames::from([
                "sparse_i64",
                "sparse_str",
                "sparse_bool",
                "sparse_f64",
                "sparse_boundary",
                "explicit_fill_values",
                "all_default",
                "clustered_edges",
                "almost_dense",
                "mixed_null_and_values",
            ]),
            vec![
                Sparse::encode(&sparse_i64_col.into_array(), None)?,
                Sparse::encode(&sparse_str_col.into_array(), None)?,
                Sparse::encode(&sparse_bool_col.into_array(), None)?,
                Sparse::encode(&sparse_f64.into_array(), None)?,
                Sparse::encode(&sparse_boundary.into_array(), None)?,
                Sparse::encode(
                    &explicit_fill_values.into_array(),
                    Some(Scalar::primitive(10i32, Nullability::Nullable)),
                )?,
                Sparse::encode(&all_default, Some(Scalar::from(10i32)))?,
                Sparse::encode(&clustered_edges.into_array(), None)?,
                Sparse::encode(
                    &almost_dense.into_array(),
                    Some(Scalar::primitive(0i32, Nullability::Nullable)),
                )?,
                Sparse::encode(&mixed_null_and_values.into_array(), Some(mixed_null_fill))?,
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
