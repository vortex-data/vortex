// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::*;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::execution::pipeline::Turn;
use crate::execution::pipeline::ValueCache;
use crate::plan::PlanRef;
use crate::test::SESSION;
use crate::test::Store;
use crate::test::dict;

fn drive_take(
    store: &Store,
    plan: &PlanRef,
    rows: std::ops::Range<u64>,
    mask: Mask,
    cache: ValueCache,
) -> VortexResult<(Vec<ArrayRef>, usize)> {
    let mut scan = Scan::try_new(SESSION.clone(), plan.clone(), vec![Split { rows, mask }])?
        .with_value_cache(cache);
    let mut inflight = Vec::new();
    let mut outputs = Vec::new();
    let mut reads = 0;
    loop {
        match scan.step()? {
            Turn::Read(req) => {
                reads += 1;
                inflight.push(req);
            }
            Turn::Output(_, array) => outputs.push(array),
            Turn::Waiting => {
                let req = inflight.remove(0);
                scan.deliver(req.id, store.read(req.segment_id))?;
            }
            Turn::Done => return Ok((outputs, reads)),
        }
    }
}

fn varbin_values(len: usize, dtype: DType) -> ArrayRef {
    VarBinArray::from_iter(
        (0..len).map(|i| (i != 1).then(|| format!("val-{i}"))),
        dtype,
    )
    .into_array()
}

#[test]
fn sparse_dict_threshold_boundaries() -> VortexResult<()> {
    let cases = [
        // Below 1024 values threshold: stays Shared even with 1 code.
        (1023, 1, DType::Utf8(Nullability::Nullable), false),
        // At 1024 values, codes * 16 == 1024 (not strictly <): stays Shared.
        (1024, 64, DType::Utf8(Nullability::Nullable), false),
        // At 1024 values, codes * 16 == 1008 < 1024: unwraps Shared to raw encoded VarBin.
        (1024, 63, DType::Utf8(Nullability::Nullable), true),
        (1024, 63, DType::Binary(Nullability::Nullable), true),
    ];

    for (values_len, codes_len, dtype, expect_unwrapped) in cases {
        let store = Store::default();
        let values = varbin_values(values_len, dtype.clone());
        let code_values = (0..codes_len)
            .map(|i| Ok((i % 7 != 0).then_some(u32::try_from(i % values_len)?)))
            .collect::<VortexResult<Vec<_>>>()?;
        let codes = PrimitiveArray::from_option_iter(code_values).into_array();
        let plan = dict(store.flat(&values)?, store.flat(&codes)?);

        let (outputs, reads) = drive_take(
            &store,
            &plan,
            0..codes_len as u64,
            Mask::new_true(codes_len),
            ValueCache::default(),
        )?;
        assert_eq!(reads, 2);
        assert_eq!(outputs.len(), 1);

        let dict_values = outputs[0].as_::<Dict>().values();
        if expect_unwrapped {
            assert!(
                dict_values.is::<VarBin>(),
                "expected raw encoded VarBin values, got {}",
                dict_values.encoding_id()
            );
            assert!(!dict_values.is::<Shared>());
        } else {
            assert!(
                dict_values.is::<Shared>(),
                "expected Shared values, got {}",
                dict_values.encoding_id()
            );
        }

        let expected = VarBinArray::from_iter(
            (0..codes_len).map(|i| {
                let idx = i % values_len;
                (i % 7 != 0 && idx != 1).then(|| format!("val-{idx}"))
            }),
            dtype,
        )
        .into_array();
        assert_arrays_eq!(outputs[0], expected, &mut SESSION.create_execution_ctx());
    }
    Ok(())
}

#[test]
fn sparse_primitive_dict_keeps_shared_values() -> VortexResult<()> {
    let store = Store::default();
    let values = PrimitiveArray::from_iter(0..1024i32).into_array();
    let codes = PrimitiveArray::from_option_iter([Some(1023u32), None, Some(5)]).into_array();
    let plan = dict(store.flat(&values)?, store.flat(&codes)?);

    let (outputs, _) = drive_take(
        &store,
        &plan,
        0..3,
        Mask::new_true(3),
        ValueCache::default(),
    )?;
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].as_::<Dict>().values().is::<Shared>());
    assert_arrays_eq!(
        outputs[0],
        PrimitiveArray::from_option_iter([Some(1023i32), None, Some(5)]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn empty_selection_reads_nothing_and_emits_no_batches() -> VortexResult<()> {
    let store = Store::default();
    let values = varbin_values(1024, DType::Utf8(Nullability::Nullable));
    let codes = PrimitiveArray::from_iter([0u32, 1, 2, 3]).into_array();
    let plan = dict(store.flat(&values)?, store.flat(&codes)?);

    let (outputs, reads) = drive_take(
        &store,
        &plan,
        0..4,
        Mask::new_false(4),
        ValueCache::default(),
    )?;
    assert_eq!(reads, 0);
    assert!(outputs.is_empty());
    Ok(())
}

#[test]
fn mixed_sparse_and_dense_chunks_share_cached_values() -> VortexResult<()> {
    let store = Store::default();
    let dtype = DType::Utf8(Nullability::Nullable);
    let values = varbin_values(1024, dtype.clone());
    // Chunk 0 has 4 codes (sparse: 4 * 16 < 1024); chunk 1 has 64 codes (dense: 64 * 16 == 1024).
    let codes = PrimitiveArray::from_option_iter((0..68u32).map(|i| (i != 2).then_some(i % 1024)))
        .into_array();
    let plan = dict(store.flat(&values)?, store.chunked(&codes, &[4, 64])?);
    let cache = ValueCache::default();

    let (sparse_out, first_reads) =
        drive_take(&store, &plan, 0..4, Mask::new_true(4), cache.clone())?;
    assert_eq!(first_reads, 2, "values + sparse codes chunk");
    assert_eq!(sparse_out.len(), 1);
    assert!(sparse_out[0].as_::<Dict>().values().is::<VarBin>());

    let (dense_out, second_reads) = drive_take(&store, &plan, 4..68, Mask::new_true(64), cache)?;
    assert_eq!(
        second_reads, 1,
        "values reused from cache; only dense codes read"
    );
    assert_eq!(dense_out.len(), 1);
    assert!(dense_out[0].as_::<Dict>().values().is::<Shared>());

    let combined = ChunkedArray::try_new(
        vec![sparse_out[0].clone(), dense_out[0].clone()],
        dtype.clone(),
    )?
    .into_array();
    let expected = VarBinArray::from_iter(
        (0..68).map(|i| (i != 1 && i != 2).then(|| format!("val-{i}"))),
        dtype,
    )
    .into_array();
    assert_arrays_eq!(combined, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn many_value_chunks_exceeding_default_capacity_join_without_deadlock() -> VortexResult<()> {
    let store = Store::default();
    let chunks = DEFAULT_CAPACITY + 4;
    let values = PrimitiveArray::from_iter(0..i32::try_from(chunks)?).into_array();
    let sizes = vec![1usize; chunks];
    let codes = PrimitiveArray::from_iter([0u32, u32::try_from(chunks - 1)?]).into_array();
    let plan = dict(store.chunked(&values, &sizes)?, store.flat(&codes)?);

    assert_eq!(TakeSource::capacity(CODES), DEFAULT_CAPACITY);
    assert_eq!(TakeSource::capacity(VALUES), UNBOUNDED);

    let (outputs, reads) = drive_take(
        &store,
        &plan,
        0..2,
        Mask::new_true(2),
        ValueCache::default(),
    )?;
    assert_eq!(reads, chunks + 1);
    if outputs.len() != 1 {
        vortex_bail!("expected one output batch, got {}", outputs.len());
    }
    assert_arrays_eq!(
        outputs[0],
        PrimitiveArray::from_iter([0i32, i32::try_from(chunks - 1)?]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}
