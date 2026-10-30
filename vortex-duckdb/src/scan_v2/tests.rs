// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rstest::rstest;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::assert_arrays_eq;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::gt;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::planning::driver::Run;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use super::FileScan;
use super::part_ranges;
use crate::RUNTIME;
use crate::SESSION;

#[rstest]
#[case::whole(0..30, &[0, 10, 20, 30], vec![0..10, 10..20, 20..30])]
#[case::clipped(5..25, &[0, 10, 20, 30], vec![5..10, 10..20, 20..25])]
#[case::inside_one_split(12..18, &[0, 10, 20, 30], vec![12..18])]
#[case::empty(5..5, &[0, 10], vec![])]
fn parts_follow_splits(
    #[case] rows: Range<u64>,
    #[case] splits: &[u64],
    #[case] expected: Vec<Range<u64>>,
) {
    assert_eq!(part_ranges(rows, splits), expected);
}

#[rstest]
#[case::disabled(0)]
#[case::prefix_and_bulk(2)]
#[case::entire_file_initial(16)]
fn reserved_work_returns_selected_rows_once(
    #[case] initial: usize,
    #[values(false, true)] sparse: bool,
    #[values(false, true)] dynamic: bool,
) -> VortexResult<()> {
    let array = (0..100_000i64).collect::<PrimitiveArray>().into_array();
    let file = RUNTIME.block_on(async {
        let mut bytes = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut bytes, array.to_array_stream())
            .await?;
        SESSION.open_options().open_buffer(bytes)
    })?;
    let filter = gt(root(), lit(5000i64)).bind(file.dtype())?;
    let mut builder = ScanBuilder::new(SESSION.clone(), file.scan_file())
        .with_some_filter(Some(filter))
        .with_row_range(100..99_900);
    if sparse {
        builder = builder.with_row_indices(StrictSortedBuffer::try_new(
            (0..100_000u64).step_by(997).collect(),
        )?);
    }
    let scan = Arc::new(FileScan::try_new(
        builder,
        &[0, 25_000, 50_000, 75_000, 100_000],
        dynamic,
        &AtomicUsize::new(initial),
        None,
        "test",
    )?);
    // Reserve every assignment before workers execute, as DuckDB can under its global lock.
    let mut reserved = Vec::new();
    while let Some(work) = scan.take_work(true) {
        reserved.push(work);
    }
    assert!(scan.take_work(true).is_none());
    assert_eq!(scan.progress(), 100.0);
    if !sparse && initial == 0 {
        assert_eq!(reserved.len(), 4);
    }
    let mut actual = std::thread::scope(|scope| -> VortexResult<Vec<i64>> {
        let handles: Vec<_> = reserved
            .into_iter()
            .map(|work| {
                let scan = Arc::clone(&scan);
                scope.spawn(move || -> VortexResult<Vec<i64>> {
                    let mut run = Run::new();
                    let mut initial_roots = Vec::new();
                    let mut values = Vec::new();
                    let mut ctx = SESSION.create_execution_ctx();
                    let mut reserved = Some(work);
                    while let Some(array) =
                        scan.next_array(&mut run, &mut initial_roots, reserved.take())?
                    {
                        values.extend_from_slice(
                            array
                                .execute::<Canonical>(&mut ctx)?
                                .into_primitive()
                                .as_slice::<i64>(),
                        );
                    }
                    Ok(values)
                })
            })
            .collect();
        let mut values = Vec::new();
        for handle in handles {
            values.extend(
                handle
                    .join()
                    .map_err(|_| vortex_err!("scan worker failed"))??,
            );
        }
        Ok(values)
    })?;
    actual.sort_unstable();
    let expected = (5001..99_900i64)
        .filter(|value| !sparse || value % 997 == 0)
        .collect::<PrimitiveArray>();
    assert_arrays_eq!(
        PrimitiveArray::from_iter(actual),
        expected,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}
