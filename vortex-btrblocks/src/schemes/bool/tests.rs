// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::assert_arrays_eq;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const LEN: usize = 4096;

fn nullable_array(values: impl Iterator<Item = Option<bool>>) -> BoolArray {
    let values: Vec<_> = values.collect();
    BoolArray::new(
        BitBuffer::from_iter(values.iter().map(|v| v.unwrap_or(false))),
        Validity::from_iter(values.iter().map(Option::is_some)),
    )
}

#[rstest]
#[case::rare_true(
    BoolArray::new(BitBuffer::from_iter((0..LEN).map(|i| i % 500 == 0)), Validity::NonNullable),
    Scalar::from(false),
)]
#[case::rare_false(
    BoolArray::new(BitBuffer::from_iter((0..LEN).map(|i| i % 500 != 7)), Validity::NonNullable),
    Scalar::from(true),
)]
#[case::rare_nulls(
    nullable_array((0..LEN).map(|i| (i % 300 != 0).then_some(true))),
    Scalar::from(Some(true)),
)]
#[case::rare_nulls_and_minority(
    nullable_array((0..LEN).map(|i| (i % 300 != 0).then_some(i % 700 != 1))),
    Scalar::from(Some(true)),
)]
#[case::mostly_null(
    nullable_array((0..LEN).map(|i| (i % 200 == 0).then_some(i % 400 == 0))),
    Scalar::null_native::<bool>(),
)]
fn test_sparse_compressed(#[case] array: BoolArray, #[case] fill: Scalar) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressor::default().compress(&array.clone().into_array(), &mut ctx)?;

    let sparse = compressed
        .as_opt::<Sparse>()
        .unwrap_or_else(|| panic!("expected sparse, got {}", compressed.encoding_id()));
    assert_eq!(sparse.fill_scalar(), &fill);
    assert!(compressed.nbytes() < array.clone().into_array().nbytes());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case::dense(BoolArray::new(
    BitBuffer::from_iter((0..LEN).map(|i| i % 3 == 0)),
    Validity::NonNullable,
))]
#[case::dense_nulls(nullable_array((0..LEN).map(|i| (i % 3 != 0).then_some(true))))]
fn test_dense_stays_canonical(#[case] array: BoolArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = BtrBlocksCompressor::default().compress(&array.clone().into_array(), &mut ctx)?;
    assert!(compressed.is::<Bool>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}
