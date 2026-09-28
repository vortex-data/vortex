// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::Expression;
use vortex_array::expr::checked_add;
use vortex_array::expr::col;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::request::IoRequest;
use vortex_io::request::IoResult;
use vortex_io::request::IoTarget;
use vortex_io::runtime::BlockingRuntime;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use super::FilterProject;
use crate::planning::OpenedFile;
use crate::planning::tests::fixtures::RUNTIME;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::concat;
use crate::planning::tests::fixtures::ctx;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::reference_scan;
use crate::planning::tests::fixtures::write_chunked_test_file;
use crate::planning::tests::fixtures::write_test_file;

fn chunks() -> Vec<ArrayRef> {
    vec![
        buffer![1u32, 2, 3, 4].into_array(),
        buffer![5u32, 6, 7, 8].into_array(),
    ]
}

fn two_chunk_file() -> VortexResult<ByteBuffer> {
    write_chunked_test_file(&[("numbers", chunks())])
}

/// A file written with the default strategy: compressed, with statistics and a string column.
fn default_file() -> VortexResult<ByteBuffer> {
    write_test_file(&[
        ("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array()),
        (
            "strings",
            VarBinArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h"]).into_array(),
        ),
    ])
}

fn stage(
    buffer: &ByteBuffer,
    filter: Option<Expression>,
    projection: Expression,
) -> VortexResult<FilterProject> {
    let footer = open_buffer(buffer)?.footer().clone();
    Ok(FilterProject::new(
        OpenedFile {
            read: Arc::new(buffer.clone()),
            size: buffer.len() as u64,
            footer,
        },
        filter,
        projection,
        SESSION.clone(),
    ))
}

/// Drives the planner to `Done`, collecting every emitted morsel with its scope.
fn morsels(planner: &mut FilterProject) -> VortexResult<Vec<(WorkScope, Box<dyn Morsel>)>> {
    let mut out = Vec::new();
    loop {
        match planner.state() {
            State::Done => return Ok(out),
            State::NeedsIO(batch) => vortex_bail!("unexpected IO {batch:?}"),
            State::NeedsCompute => match planner.compute()? {
                PlannerOutput::Done => return Ok(out),
                PlannerOutput::Continue => continue,
                PlannerOutput::Morsel(scope, morsel) => out.push((scope, morsel)),
                other => vortex_bail!("unexpected output {other:?}"),
            },
        }
    }
}

fn read_range(buffer: &ByteBuffer, request: &IoRequest) -> VortexResult<IoResult> {
    let IoTarget::Range { offset, len } = request.target else {
        vortex_bail!("a morsel asked for {:?}", request.target);
    };
    Ok(IoResult::Bytes(RUNTIME.block_on(buffer.read_at(
        offset,
        len,
        Alignment::none(),
    ))?))
}

/// Runs one morsel to completion, answering its IO from `buffer`. Returns the batches and the
/// number of `NeedsIO` rounds observed. `reverse` delivers each batch back to front.
fn serve(
    mut morsel: Box<dyn Morsel>,
    buffer: &ByteBuffer,
    reverse: bool,
) -> VortexResult<(Vec<ArrayRef>, usize)> {
    let mut out = Vec::new();
    let mut rounds = 0;
    for _ in 0..64 {
        match morsel.state() {
            State::Done => return Ok((out, rounds)),
            State::NeedsIO(mut batch) => {
                rounds += 1;
                if reverse {
                    batch.reverse();
                }
                for request in &batch {
                    morsel.set_io_result(request.request, read_range(buffer, request)?);
                }
            }
            State::NeedsCompute => match morsel.compute()? {
                MorselOutput::Done => return Ok((out, rounds)),
                MorselOutput::Continue | MorselOutput::NeedsIO(_) => continue,
                MorselOutput::Batch(array) => out.push(array),
            },
        }
    }
    vortex_bail!("morsel did not finish within 64 visits")
}

fn missing(which: &str) -> VortexError {
    vortex_err!("missing {which} morsel")
}

#[test]
fn one_morsel_per_split_in_order() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let mut planner = stage(&buffer, None, root())?;
    let scopes: Vec<_> = morsels(&mut planner)?
        .into_iter()
        .map(|(scope, _)| scope.rows)
        .collect();
    assert_eq!(scopes, vec![0..4, 4..8]);
    Ok(())
}

#[test]
fn root_projection_returns_each_chunk_in_one_round() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let mut planner = stage(&buffer, None, root())?;
    for ((_, morsel), chunk) in morsels(&mut planner)?.into_iter().zip(chunks()) {
        let (out, rounds) = serve(morsel, &buffer, false)?;
        assert_eq!(rounds, 1, "a flat chunk discovers all its segments at once");
        assert_eq!(out.len(), 1);
        let expected = StructArray::from_fields(&[("numbers", chunk)])?;
        assert_arrays_eq!(out[0], expected, &mut ctx());
    }
    Ok(())
}

#[test]
fn filter_across_the_chunk_boundary() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let mut planner = stage(&buffer, Some(gt(col("numbers"), lit(6u32))), root())?;
    let mut morsels = morsels(&mut planner)?.into_iter();
    let (_, first) = morsels.next().ok_or_else(|| missing("first"))?;
    assert!(serve(first, &buffer, false)?.0.is_empty());
    let (_, second) = morsels.next().ok_or_else(|| missing("second"))?;
    let (out, _) = serve(second, &buffer, false)?;
    assert_eq!(out.len(), 1);
    let expected = StructArray::from_fields(&[("numbers", buffer![7u32, 8].into_array())])?;
    assert_arrays_eq!(out[0], expected, &mut ctx());
    Ok(())
}

#[test]
fn filter_matching_nothing_yields_no_batches() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let mut planner = stage(&buffer, Some(gt(col("numbers"), lit(100u32))), root())?;
    let morsels = morsels(&mut planner)?;
    assert_eq!(morsels.len(), 2);
    for (_, morsel) in morsels {
        assert!(serve(morsel, &buffer, false)?.0.is_empty());
    }
    Ok(())
}

#[test]
fn expression_projection_matches_the_existing_scan() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let filter = gt(col("numbers"), lit(2u32));
    let projection = checked_add(col("numbers"), lit(1u32));
    let mut planner = stage(&buffer, Some(filter.clone()), projection.clone())?;
    let mut actual = Vec::new();
    for (_, morsel) in morsels(&mut planner)? {
        actual.extend(serve(morsel, &buffer, false)?.0);
    }
    let expected = reference_scan(&buffer, Some(filter), projection)?;
    assert_eq!(actual.len(), 2);
    let dtype = expected
        .first()
        .map(|array| array.dtype().clone())
        .ok_or_else(|| missing("reference"))?;
    assert_arrays_eq!(
        concat(actual, &dtype)?,
        concat(expected, &dtype)?,
        &mut ctx()
    );
    Ok(())
}

#[test]
fn unbound_projection_column_fails_on_first_compute() -> VortexResult<()> {
    let buffer = two_chunk_file()?;
    let mut planner = stage(&buffer, None, col("missing"))?;
    let err = planner.compute().err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("missing")),
        "{err:?}"
    );
    Ok(())
}

/// A default-strategy file has compressed leaves, statistics, and a string column, so the
/// reader's future is deeper than one flat segment. It must still complete purely through
/// protocol deliveries, and delivery order within a batch must not matter.
#[rstest]
#[case::in_order(false)]
#[case::reversed(true)]
fn default_file_completes_through_the_protocol(#[case] reverse: bool) -> VortexResult<()> {
    let buffer = default_file()?;
    let filter = gt(col("numbers"), lit(3u32));
    let mut planner = stage(&buffer, Some(filter.clone()), root())?;
    let morsels = morsels(&mut planner)?;
    assert_eq!(morsels.len(), 1);
    let mut actual = Vec::new();
    let mut rounds = 0;
    for (_, morsel) in morsels {
        let (out, morsel_rounds) = serve(morsel, &buffer, reverse)?;
        actual.extend(out);
        rounds += morsel_rounds;
    }
    assert!(rounds >= 1, "data must have come through NeedsIO");
    let expected = reference_scan(&buffer, Some(filter), root())?;
    let dtype = expected[0].dtype().clone();
    assert_arrays_eq!(
        concat(actual, &dtype)?,
        concat(expected, &dtype)?,
        &mut ctx()
    );
    Ok(())
}

#[test]
fn partial_delivery_keeps_the_rest_outstanding() -> VortexResult<()> {
    let buffer = default_file()?;
    let mut planner = stage(&buffer, None, root())?;
    let (_, mut morsel) = morsels(&mut planner)?
        .pop()
        .ok_or_else(|| missing("only"))?;
    assert_eq!(morsel.state(), State::NeedsCompute);
    let MorselOutput::NeedsIO(batch) = morsel.compute()? else {
        vortex_bail!("expected the first poll to miss");
    };
    assert!(
        batch.len() >= 2,
        "two columns need at least two segments: {batch:?}"
    );
    assert_eq!(morsel.state(), State::NeedsIO(batch.clone()));

    morsel.set_io_result(batch[0].request, read_range(&buffer, &batch[0])?);
    assert_eq!(morsel.state(), State::NeedsIO(batch[1..].to_vec()));

    for request in &batch[1..] {
        morsel.set_io_result(request.request, read_range(&buffer, request)?);
    }
    assert_eq!(morsel.state(), State::NeedsCompute);
    let (out, _) = serve(morsel, &buffer, false)?;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 8);
    Ok(())
}
