// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::arrays::shared::Shared;
use crate::arrays::shared::SharedArrayExt;
use crate::arrays::shared::SharedData;
use crate::arrays::shared::initialize;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn shared_array_caches_on_canonicalize() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
    let shared = Shared::new(array);

    let mut ctx = SESSION.create_execution_ctx();

    let first = shared.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

    // Second call should return cached without invoking the closure.
    let second = shared.get_or_compute(|_| panic!("should not execute twice"))?;

    assert!(ArrayRef::ptr_eq(&first, &second));
    assert_arrays_eq!(first, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);

    Ok(())
}

/// A `LIKE` over a share of an FSST array runs the FSST kernel on the source and leaves the share
/// uncomputed.
#[test]
fn like_runs_on_encoded_source() -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_fsst::initialize(&session);
    initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let strings = VarBinArray::from_iter(
        [
            Some("http://google.com"),
            None,
            Some("yandex.ru"),
            Some("maps.google.de"),
        ],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let compressor = vortex_fsst::fsst_train_compressor(&strings, &mut ctx)?;
    let fsst = vortex_fsst::fsst_compress(&strings, &compressor, &mut ctx)?.into_array();

    let shared = Shared::new(fsst);
    let pattern = ConstantArray::new("%google%", shared.len()).into_array();
    let like = Like::try_new(shared.clone().into_array(), pattern, LikeOptions::default())?;

    let result = like
        .into_array()
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true), None, Some(false), Some(true)]),
        &mut ctx
    );

    let data: &SharedData = shared.data();
    assert!(!data.is_computed(), "the LIKE canonicalized the share");
    Ok(())
}

/// A dictionary that reads few of its shared FSST values takes them from the encoded source and
/// leaves the share uncomputed.
#[test]
fn sparse_take_reads_encoded_source() -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_fsst::initialize(&session);
    initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let values: Vec<String> = (0..2048)
        .map(|i| format!("http://site{i}.example.org/"))
        .collect();
    let strings = VarBinArray::from_iter(
        values.iter().map(|value| Some(value.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let compressor = vortex_fsst::fsst_train_compressor(&strings, &mut ctx)?;
    let fsst = vortex_fsst::fsst_compress(&strings, &compressor, &mut ctx)?.into_array();

    let shared = Shared::new(fsst);
    let codes = PrimitiveArray::from_iter([7u16, 2000, 7]).into_array();
    let dict = DictArray::try_new(codes, shared.clone().into_array())?;

    let result = dict
        .into_array()
        .execute::<Canonical>(&mut ctx)?
        .into_array();
    let expected = VarBinArray::from_iter(
        [&values[7], &values[2000], &values[7]].map(|value| Some(value.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    assert_arrays_eq!(result, expected, &mut ctx);

    let data: &SharedData = shared.data();
    assert!(!data.is_computed(), "the take canonicalized the share");
    Ok(())
}
