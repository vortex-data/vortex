// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::check_metadata;
use vortex_session::VortexSession;

use crate::OnPair;
use crate::OnPairMetadata;
use crate::compress::DEFAULT_DICT12_CONFIG;
use crate::compress::onpair_compress;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn sample_input() -> VarBinArray {
    VarBinArray::from_iter(
        [
            Some("https://www.example.com/page"),
            Some("https://www.example.com/data"),
            Some("https://www.test.org/page"),
            Some("ftp://files.example.com/x"),
            Some("https://www.example.com/page"),
        ],
        DType::Utf8(Nullability::NonNullable),
    )
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_metadata_golden() {
    check_metadata(
        "onpair.metadata",
        &OnPairMetadata {
            uncompressed_lengths_ptype: PType::I32 as i32,
            bits: 12,
        }
        .encode_to_vec(),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_roundtrip() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();

    let compressed = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).expect("compress");
    assert!(compressed.clone().into_array().is::<OnPair>());

    let mut ctx = SESSION.create_execution_ctx();
    let decoded = compressed
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .expect("canonicalize");

    decoded
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), 5);
            assert_eq!(
                got[0].as_deref(),
                Some(b"https://www.example.com/page".as_ref())
            );
            assert_eq!(
                got[3].as_deref(),
                Some(b"ftp://files.example.com/x".as_ref())
            );
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_nullable_canonicalize() {
    let input = VarBinArray::from_iter(
        [Some("a"), None, Some("bbb"), None, Some("ccccc")],
        DType::Utf8(Nullability::Nullable),
    );
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = arr
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    canonical
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got[1], None);
            assert_eq!(got[3], None);
            assert_eq!(got[4].as_deref(), Some(b"ccccc".as_ref()));
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_scalar_at() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    let s = arr.into_array().execute_scalar(2, &mut ctx).unwrap();
    let v = s.as_utf8().value().unwrap();
    assert_eq!(v.as_bytes(), b"https://www.test.org/page");
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_eq_pushdown() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();
    let mut ctx = SESSION.create_execution_ctx();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG)
        .unwrap()
        .into_array();

    let rhs = ConstantArray::new("https://www.example.com/page", arr.len()).into_array();
    let eq = arr
        .binary(rhs, Operator::Eq)
        .unwrap()
        .execute::<vortex_array::Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    assert_eq!(eq.as_bool_typed().true_count().unwrap(), 2);
}

fn run_like(arr: &vortex_array::ArrayRef, pattern: &str) -> vortex_array::ArrayRef {
    let n = arr.len();
    let pat = ConstantArray::new(pattern, n).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    Like.try_new_array(n, LikeOptions::default(), [arr.clone(), pat])
        .unwrap()
        .into_array()
        .execute::<vortex_array::Canonical>(&mut ctx)
        .unwrap()
        .into_array()
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_like_prefix() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG)
        .unwrap()
        .into_array();
    let result = run_like(&arr, "https://www.%");
    assert_eq!(result.as_bool_typed().true_count().unwrap(), 4);
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_like_contains() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG)
        .unwrap()
        .into_array();
    let result = run_like(&arr, "%example.com%");
    assert_eq!(result.as_bool_typed().true_count().unwrap(), 4);
}
