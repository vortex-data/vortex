// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressor behaviour over the default scheme set.

use std::sync::LazyLock;

use rstest::rstest;
#[cfg(feature = "zstd")]
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
#[cfg(feature = "zstd")]
use crate::COMPACT_SCHEMES;
use crate::DEFAULT_SCHEMES;
#[cfg(feature = "zstd")]
use crate::Scheme;
#[cfg(feature = "zstd")]
use crate::schemes::binary;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[rstest]
#[case::zctl(
    unsafe {
        ListViewArray::new_unchecked(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0i32, 3].into_array(),
            buffer![3i32, 2].into_array(),
            Validity::NonNullable,
        ).with_zero_copy_to_list(true)
    },
    true,
)]
#[case::overlapping(
    ListViewArray::new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0i32, 0, 0].into_array(),
        buffer![3i32, 3, 3].into_array(),
        Validity::NonNullable,
    ),
    false,
)]
fn listview_compress_roundtrip(
    #[case] input: ListViewArray,
    #[case] expect_list: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array_ref = input.clone().into_array();
    let result = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec())
        .compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    if expect_list {
        assert!(result.as_opt::<List>().is_some());
    } else {
        assert!(result.as_opt::<ListView>().is_some());
    }
    assert_arrays_eq!(result, input, &mut ctx);
    Ok(())
}

#[test]
fn test_constant_all_true() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BoolArray::new(BitBuffer::from(vec![true; 100]), Validity::NonNullable);
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_constant_all_false() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BoolArray::new(BitBuffer::from(vec![false; 100]), Validity::NonNullable);
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_nullable_all_valid_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BoolArray::new(
        BitBuffer::from(vec![true; 100]),
        Validity::from(BitBuffer::from(vec![true; 100])),
    );
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_nullable_with_nulls_not_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::from(BitBuffer::from_iter((0..100).map(|i| i % 3 != 0)));
    let array = BoolArray::new(BitBuffer::from(vec![true; 100]), validity);
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(!compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_mixed_not_constant() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = BoolArray::new(
        BitBuffer::from(vec![true, false, true, false, true]),
        Validity::NonNullable,
    );
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(!compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_binary_constant_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = vec![Some(b"constant-bytes".as_slice()); 100];
    let array = VarBinViewArray::from_iter(values, DType::Binary(Nullability::NonNullable));
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Constant>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[test]
fn test_binary_dict_compressed() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let distinct_values: [&[u8]; 3] = [b"alpha", b"beta", b"gamma"];
    let values = (0..1000)
        .map(|idx| Some(distinct_values[idx % distinct_values.len()]))
        .collect::<Vec<_>>();
    let array = VarBinViewArray::from_iter(values, DType::Binary(Nullability::NonNullable));
    let btr = BtrBlocksCompressor::new(DEFAULT_SCHEMES.to_vec());
    let compressed = btr.compress(
        &array.clone().into_array(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(compressed.is::<Dict>());
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

#[cfg(feature = "zstd")]
#[test]
fn test_compact_binary_zstd_compressed() -> VortexResult<()> {
    let values = (0..1024)
        .map(|idx| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&(idx as u32).to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect::<Vec<_>>();
    let array = VarBinViewArray::from_iter(
        values.iter().map(|value| Some(value.as_slice())),
        DType::Binary(Nullability::NonNullable),
    );

    let compressor = BtrBlocksCompressor::new(
        DEFAULT_SCHEMES
            .iter()
            .copied()
            .chain(COMPACT_SCHEMES.iter().copied())
            .collect(),
    );
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;

    assert!(
        compressed.is::<vortex_zstd::Zstd>(),
        "expected Zstd, got {}",
        compressed.encoding_id()
    );
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}

/// Each binary Zstd scheme writes exactly its own encoding, so a compressor over just that
/// scheme emits it.
#[cfg(feature = "zstd")]
#[rstest]
#[case::array_level(&binary::ZstdScheme, vortex_zstd::Zstd.id())]
#[case::buffer_level(&binary::ZstdBuffersScheme, vortex_zstd::ZstdBuffers.id())]
fn test_binary_zstd_scheme_encoding(
    #[case] scheme: &'static dyn Scheme,
    #[case] expected: ArrayId,
) -> VortexResult<()> {
    let values = (0..1024)
        .map(|idx| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&(idx as u32).to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect::<Vec<_>>();
    let array = VarBinViewArray::from_iter(
        values.iter().map(|value| Some(value.as_slice())),
        DType::Binary(Nullability::NonNullable),
    );

    let compressor = BtrBlocksCompressor::new(vec![scheme]);
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array.clone().into_array(), &mut ctx)?;

    assert_eq!(compressed.encoding_id(), expected);
    assert_arrays_eq!(compressed, array, &mut ctx);
    Ok(())
}
