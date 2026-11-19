// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `Sparse` fixture for arrays whose patches carry chunk offsets, written as `vortex.sparse.v2`.

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::encodings::sparse::Sparse;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// Several 1024-row patch chunks, the last one partial, so the chunk offsets have more than one
/// entry.
const LEN: usize = 4 * N + 7;

/// Compatibility coverage for sparse arrays with patch chunk offsets.
pub struct SparseV2Fixture;

impl FlatLayoutFixture for SparseV2Fixture {
    fn name(&self) -> &str {
        "sparse_v2.vortex"
    }

    fn description(&self) -> &str {
        "Sparse arrays whose patches carry per-chunk offsets across several 1024-row chunks"
    }

    /// The in-memory encoding: the v2 wire ID only appears on serialization.
    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    fn build(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        // Patches spread over every chunk.
        let scattered_i64 = PrimitiveArray::from_option_iter(
            (0..LEN as i64).map(|i| (i % 37 == 0).then_some(i * 1000)),
        );

        // Patches either side of each chunk boundary and in the partial last chunk, with the
        // third chunk left empty so consecutive offsets repeat.
        let boundaries = PrimitiveArray::from_option_iter((0..LEN as i64).map(|i| {
            let in_chunk = i % N as i64;
            let chunk = i / N as i64;
            (chunk != 2 && (in_chunk == 0 || in_chunk == N as i64 - 1 || i == LEN as i64 - 1))
                .then_some(i * 7)
        }));

        // A non-null fill value with a few exceptions.
        let explicit_fill =
            PrimitiveArray::from_iter((0..LEN as i32).map(|i| if i % 300 == 0 { 99 } else { 10 }));

        let sparse_str: Vec<Option<&str>> = (0..LEN)
            .map(|i| (i % 101 == 0).then_some("rare_value"))
            .collect();
        let sparse_str_col = VarBinArray::from_nullable_strs(sparse_str);

        let sparse_bool = BoolArray::from_iter((0..LEN).map(|i| (i % 211 == 0).then_some(true)));

        let arr = StructArray::try_new(
            FieldNames::from([
                "scattered_i64",
                "chunk_boundaries",
                "explicit_fill",
                "sparse_str",
                "sparse_bool",
            ]),
            vec![
                Sparse::encode_with_chunk_offsets(&scattered_i64.into_array(), None, ctx)?,
                Sparse::encode_with_chunk_offsets(&boundaries.into_array(), None, ctx)?,
                Sparse::encode_with_chunk_offsets(
                    &explicit_fill.into_array(),
                    Some(Scalar::from(10i32)),
                    ctx,
                )?,
                Sparse::encode_with_chunk_offsets(&sparse_str_col.into_array(), None, ctx)?,
                Sparse::encode_with_chunk_offsets(&sparse_bool.into_array(), None, ctx)?,
            ],
            LEN,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
