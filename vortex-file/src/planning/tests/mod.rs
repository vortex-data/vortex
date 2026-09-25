// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end footer planning tests with diagnostic range output.

pub mod fixtures;
mod range_morsel;

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::Expression;
use vortex_array::expr::checked_add;
use vortex_array::expr::col;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::ReadAtIoSource;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Driver;
use vortex_scan::planning::next::PendingPlanner;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::next::pending;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::Footer;
use crate::planning::FileSource;
use crate::planning::plan_file;
use crate::planning::plan_scan;
use crate::planning::tests::fixtures::FailingReadAt;
use crate::planning::tests::fixtures::LifoReadAtIoSource;
use crate::planning::tests::fixtures::PanickingReadAt;
use crate::planning::tests::fixtures::RUNTIME;
use crate::planning::tests::fixtures::RecordingReadAt;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::concat;
use crate::planning::tests::fixtures::ctx;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::reference_scan;
use crate::planning::tests::fixtures::write_chunked_test_file;
use crate::planning::tests::fixtures::write_test_file;
use crate::planning::tests::range_morsel::RangeOnly;

/// Runs `plan_file` through the driver with a read-at IO source.
fn run(
    read: Arc<dyn VortexReadAt>,
    size: Option<u64>,
    footer: Option<Footer>,
    filter: Option<Expression>,
) -> VortexResult<Vec<Batch>> {
    let root = plan_file(
        FileSource {
            read: Arc::clone(&read),
            size,
            footer,
        },
        filter,
        SESSION.clone(),
        next_fn(|opened| Ok(RangeOnly::new(opened))),
    );
    Driver::new(Arc::new(ReadAtIoSource::new(
        read,
        Arc::new(RUNTIME.clone()),
    )))
    .with_step_limit(10_000)
    .run(root)
}

fn numbers_file() -> VortexResult<ByteBuffer> {
    write_test_file(&[("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array())])
}

fn range_row(start: u64, end: u64) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("start", buffer![start].into_array()),
        ("end", buffer![end].into_array()),
    ])?
    .into_array())
}

#[test]
fn surviving_file_emits_the_file_range() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let size = buffer.len() as u64;
    let batches = run(
        Arc::new(buffer),
        Some(size),
        None,
        Some(gt(col("numbers"), lit(3u32))),
    )?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].scope.rows, 0..8);
    assert_arrays_eq!(batches[0].array, range_row(0, 8)?, &mut ctx());
    Ok(())
}

#[test]
fn pruned_file_reads_only_the_footer() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let len = buffer.len();
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let batches = run(
        Arc::clone(&recording) as Arc<dyn VortexReadAt>,
        Some(len as u64),
        None,
        Some(gt(col("numbers"), lit(100u32))),
    )?;
    assert!(batches.is_empty());
    assert_eq!(recording.reads(), vec![(0, len)]);
    assert_eq!(recording.size_calls(), 0);
    Ok(())
}

#[test]
fn cached_footer_needs_no_reads() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let footer = open_buffer(&buffer)?.footer().clone();
    let batches = run(
        Arc::new(PanickingReadAt),
        Some(buffer.len() as u64),
        Some(footer),
        None,
    )?;
    assert_eq!(batches.len(), 1);
    assert_arrays_eq!(batches[0].array, range_row(0, 8)?, &mut ctx());
    Ok(())
}

#[test]
fn failing_source_surfaces_the_error() -> VortexResult<()> {
    let size = numbers_file()?.len() as u64;
    let err = run(Arc::new(FailingReadAt(size)), None, None, None)
        .err()
        .map(|e| e.to_string());
    assert!(
        err.as_deref()
            .is_some_and(|m| m.contains(FailingReadAt::MESSAGE)),
        "{err:?}"
    );
    Ok(())
}

#[test]
fn empty_file_emits_nothing() -> VortexResult<()> {
    let buffer = write_test_file(&[("numbers", buffer![0u32; 0].into_array())])?;
    let size = buffer.len() as u64;
    let batches = run(Arc::new(buffer), Some(size), None, None)?;
    assert!(batches.is_empty());
    Ok(())
}

#[rstest]
#[case::known_size(true, 0)]
#[case::unknown_size(false, 1)]
fn known_and_unknown_size_read_only_the_footer(
    #[case] known: bool,
    #[case] size_calls: usize,
) -> VortexResult<()> {
    let buffer = numbers_file()?;
    let len = buffer.len();
    let recording = Arc::new(RecordingReadAt::new(buffer));
    let batches = run(
        Arc::clone(&recording) as Arc<dyn VortexReadAt>,
        known.then_some(len as u64),
        None,
        None,
    )?;
    assert_eq!(batches.len(), 1);
    assert_eq!(recording.size_calls(), size_calls);
    assert_eq!(recording.reads(), vec![(0, len)]);
    Ok(())
}

/// Runs `plan_scan` through the driver over `io`, with `read` as the file's own reader.
fn run_scan(
    read: Arc<dyn VortexReadAt>,
    io: Arc<dyn IoSource>,
    size: Option<u64>,
    footer: Option<Footer>,
    filter: Option<Expression>,
    projection: Expression,
) -> VortexResult<Vec<Batch>> {
    let root = plan_scan(
        FileSource { read, size, footer },
        filter,
        projection,
        SESSION.clone(),
    );
    Driver::new(io).with_step_limit(10_000).run(root)
}

fn read_at_source(buffer: &ByteBuffer) -> Arc<dyn IoSource> {
    Arc::new(ReadAtIoSource::new(
        Arc::new(buffer.clone()),
        Arc::new(RUNTIME.clone()),
    ))
}

/// With a cached footer the file's own reader is never needed: every data byte arrives through
/// the driver's IO source. A panicking reader on the file proves no stage reads around the
/// protocol.
#[test]
fn data_reads_go_only_through_the_io_source() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let footer = open_buffer(&buffer)?.footer().clone();
    let filter = gt(col("numbers"), lit(2u32));
    let batches = run_scan(
        Arc::new(PanickingReadAt),
        read_at_source(&buffer),
        Some(buffer.len() as u64),
        Some(footer),
        Some(filter.clone()),
        root(),
    )?;
    let expected = reference_scan(&buffer, Some(filter), root())?;
    let dtype = expected[0].dtype().clone();
    let actual = batches.into_iter().map(|batch| batch.array).collect();
    assert_arrays_eq!(
        concat(actual, &dtype)?,
        concat(expected, &dtype)?,
        &mut ctx()
    );
    Ok(())
}

/// One parity query: the filter and projection, and how many natural splits keep rows.
#[derive(Clone, Copy, Debug)]
enum Query {
    NoFilter,
    FilterSome,
    FilterNone,
    ExpressionProjection,
}

impl Query {
    fn filter(self) -> Option<Expression> {
        match self {
            Self::NoFilter => None,
            Self::FilterSome => Some(gt(col("numbers"), lit(4u32))),
            Self::FilterNone => Some(gt(col("numbers"), lit(100u32))),
            Self::ExpressionProjection => Some(gt(col("numbers"), lit(2u32))),
        }
    }

    fn projection(self) -> Expression {
        match self {
            Self::ExpressionProjection => checked_add(col("numbers"), lit(1u32)),
            _ => root(),
        }
    }

    /// Number of chunks of `1..=9` split `chunks` ways that keep at least one row.
    fn non_empty_splits(self, chunks: usize) -> usize {
        match (self, chunks) {
            (Self::FilterNone, _) => 0,
            (Self::NoFilter | Self::ExpressionProjection, n) => n,
            (Self::FilterSome, 1) => 1,
            (Self::FilterSome, 3) => 2,
            (query, n) => unreachable!("no expectation for {query:?} over {n} chunks"),
        }
    }
}

fn parity_file(chunks: usize) -> VortexResult<ByteBuffer> {
    let per = u32::try_from(9 / chunks)?;
    let numbers: Vec<ArrayRef> = (0..u32::try_from(chunks)?)
        .map(|i| {
            let start = i * per + 1;
            PrimitiveArray::from_iter(start..start + per).into_array()
        })
        .collect();
    if chunks == 1 {
        write_test_file(&[("numbers", numbers[0].clone())])
    } else {
        write_chunked_test_file(&[("numbers", numbers)])
    }
}

#[rstest]
fn parity_with_the_existing_scan(
    #[values(
        Query::NoFilter,
        Query::FilterSome,
        Query::FilterNone,
        Query::ExpressionProjection
    )]
    query: Query,
    #[values(1, 3)] chunks: usize,
) -> VortexResult<()> {
    let buffer = parity_file(chunks)?;
    let batches = run_scan(
        Arc::new(buffer.clone()),
        read_at_source(&buffer),
        Some(buffer.len() as u64),
        None,
        query.filter(),
        query.projection(),
    )?;
    assert_eq!(batches.len(), query.non_empty_splits(chunks));
    let expected = reference_scan(&buffer, query.filter(), query.projection())?;
    let dtype = query
        .projection()
        .bind(open_buffer(&buffer)?.dtype())?
        .dtype()
        .clone();
    let actual = batches.into_iter().map(|batch| batch.array).collect();
    assert_arrays_eq!(
        concat(actual, &dtype)?,
        concat(expected, &dtype)?,
        &mut ctx()
    );
    Ok(())
}

/// A root that hands out several pending files, one per compute, then finishes.
struct Fanout {
    files: Vec<Box<dyn PendingPlanner>>,
    ordinal: u64,
}

impl IoConsumer for Fanout {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Fanout {
    fn state(&self) -> State {
        if self.files.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let file = self.files.remove(0);
        let scope = WorkScope {
            file_ordinal: self.ordinal,
            rows: 0..0,
        };
        self.ordinal += 1;
        Ok(PlannerOutput::Planner(scope, file))
    }
}

/// Two files planned concurrently over one IO source, both with an unknown size.
fn two_files_root(buffer: &ByteBuffer) -> Box<dyn PendingPlanner> {
    let plan = |projection: Expression| {
        plan_scan(
            FileSource {
                read: Arc::new(buffer.clone()),
                size: None,
                footer: None,
            },
            None,
            projection,
            SESSION.clone(),
        )
    };
    let files = vec![
        plan(col("numbers")),
        plan(checked_add(col("numbers"), lit(100u32))),
    ];
    pending(move || Ok(Box::new(Fanout { files, ordinal: 0 }) as Box<dyn Planner>))
}

fn expect_two_files(buffer: &ByteBuffer, batches: Vec<Batch>) -> VortexResult<()> {
    assert_eq!(batches.len(), 2, "one batch per file");
    let plain = reference_scan(buffer, None, col("numbers"))?;
    let shifted = reference_scan(buffer, None, checked_add(col("numbers"), lit(100u32)))?;
    let dtype = plain[0].dtype().clone();
    let mut actual: Vec<ArrayRef> = batches.into_iter().map(|batch| batch.array).collect();
    // The order the two files finish in depends on the source; compare as a set of two.
    let first_is_plain = concat(vec![actual[0].clone()], &dtype)?
        .display_values()
        .to_string()
        == concat(plain.clone(), &dtype)?.display_values().to_string();
    if !first_is_plain {
        actual.swap(0, 1);
    }
    assert_arrays_eq!(actual[0], concat(plain, &dtype)?, &mut ctx());
    assert_arrays_eq!(actual[1], concat(shifted, &dtype)?, &mut ctx());
    Ok(())
}

/// The LIFO source completes the newest submission first, so the second file's size, tail, and
/// data all finish while the first file's size request is still waiting. Morsels are their own
/// work items, so six requests are submitted and delivered in the recorded order.
#[test]
fn two_files_complete_out_of_order() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let source = Arc::new(LifoReadAtIoSource::new(Arc::new(buffer.clone())));
    let batches = Driver::new(Arc::clone(&source) as Arc<dyn IoSource>)
        .with_step_limit(10_000)
        .run(two_files_root(&buffer))?;

    let submitted = source.submitted();
    let completed = source.completed();
    assert_eq!(submitted.len(), 6, "{submitted:?}");
    assert_eq!(completed.len(), 6, "{completed:?}");
    let (first, second) = (submitted[0], submitted[1]);
    assert_ne!(
        first, second,
        "both files had a size request in flight before either completed"
    );
    let data_of_second = submitted[3];
    let data_of_first = submitted[5];
    assert!(![first, second].contains(&data_of_second));
    assert!(![first, second, data_of_second].contains(&data_of_first));
    assert_eq!(
        submitted,
        vec![first, second, second, data_of_second, first, data_of_first]
    );
    assert_eq!(
        completed,
        vec![second, second, data_of_second, first, first, data_of_first]
    );
    expect_two_files(&buffer, batches)
}

#[test]
fn two_files_over_the_read_at_source() -> VortexResult<()> {
    let buffer = numbers_file()?;
    let batches = Driver::new(read_at_source(&buffer))
        .with_step_limit(10_000)
        .run(two_files_root(&buffer))?;
    expect_two_files(&buffer, batches)
}

/// Not a correctness test: prints wall time of the prototype against the existing scan on a
/// default-strategy file. Run with
/// `cargo nextest run --release -p vortex-file --run-ignored ignored-only --no-capture timing`.
#[test]
#[ignore]
fn timing_against_existing_scan() -> VortexResult<()> {
    const ROWS: u32 = 4_000_000;
    let ids = PrimitiveArray::from_iter(0..ROWS).into_array();
    let keys =
        PrimitiveArray::from_iter((0..ROWS).map(|i| u64::from(i) * 7919 % 1_000_003)).into_array();
    let values =
        PrimitiveArray::from_iter((0..ROWS).map(|i| f64::from(i % 1000) * 0.5)).into_array();
    let buffer = write_test_file(&[("a", ids), ("b", keys), ("c", values)])?;
    let size = buffer.len() as u64;
    eprintln!("file: {} bytes, {ROWS} rows", buffer.len());

    let queries: [(&str, Option<Expression>, Expression); 3] = [
        ("full scan", None, root()),
        (
            "filter b > 500000",
            Some(gt(col("b"), lit(500_000u64))),
            root(),
        ),
        (
            "filter + a + 1",
            Some(gt(col("b"), lit(500_000u64))),
            checked_add(col("a"), lit(1u32)),
        ),
    ];
    // Arrays come back lazy, and a canonical struct only materialises its shell, so decode each
    // batch and every field of it inside the timer. Returns the rows of the leaf columns.
    fn materialize(array: ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> VortexResult<usize> {
        use vortex_array::arrays::struct_::StructArrayExt;
        match array.execute::<vortex_array::Canonical>(ctx)? {
            vortex_array::Canonical::Struct(fields) => {
                let n = fields
                    .dtype()
                    .as_struct_fields_opt()
                    .map_or(0, |f| f.nfields());
                (0..n)
                    .map(|i| materialize(fields.unmasked_field(i).clone(), ctx))
                    .sum()
            }
            other => Ok(other.into_array().len()),
        }
    }
    let decoded = |arrays: Vec<ArrayRef>| -> VortexResult<(usize, usize)> {
        let mut ctx = SESSION.create_execution_ctx();
        let batches = arrays.len();
        let rows = arrays
            .into_iter()
            .map(|array| materialize(array, &mut ctx))
            .sum::<VortexResult<usize>>()?;
        Ok((batches, rows))
    };
    for (name, filter, projection) in queries {
        for round in 0..3 {
            let start = std::time::Instant::now();
            let (reference_batches, reference) =
                decoded(reference_scan(&buffer, filter.clone(), projection.clone())?)?;
            let reference_ms = start.elapsed().as_secs_f64() * 1e3;

            let start = std::time::Instant::now();
            let batches = run_scan(
                Arc::new(buffer.clone()),
                read_at_source(&buffer),
                Some(size),
                None,
                filter.clone(),
                projection.clone(),
            )?;
            let (prototype_batches, prototype) =
                decoded(batches.into_iter().map(|batch| batch.array).collect())?;
            let prototype_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(reference, prototype);
            eprintln!(
                "{name:<20} round {round}: existing {reference_ms:8.1} ms ({reference_batches} batches)  prototype {prototype_ms:8.1} ms ({prototype_batches} batches)  {prototype} rows"
            );
        }
    }
    Ok(())
}
