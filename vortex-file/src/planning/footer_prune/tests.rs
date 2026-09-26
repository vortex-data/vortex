// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::expr::col;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;

use super::*;
use crate::planning::tests::fixtures::SESSION;
use crate::planning::tests::fixtures::open_buffer;
use crate::planning::tests::fixtures::recording_child;
use crate::planning::tests::fixtures::write_no_stats_test_file;
use crate::planning::tests::fixtures::write_test_file;

fn columns() -> Vec<(&'static str, vortex_array::ArrayRef)> {
    vec![
        ("numbers", buffer![1u32, 2, 3, 4, 5, 6, 7, 8].into_array()),
        (
            "strings",
            VarBinArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h"]).into_array(),
        ),
    ]
}

fn opened(buffer: ByteBuffer) -> VortexResult<OpenedFile> {
    let footer = open_buffer(&buffer)?.footer().clone();
    Ok(OpenedFile {
        size: buffer.len() as u64,
        read: Arc::new(buffer),
        footer,
    })
}

fn prune(
    buffer: ByteBuffer,
    filter: Option<Expression>,
) -> VortexResult<(FooterPrune, Arc<AtomicBool>)> {
    let (next, invoked) = recording_child();
    Ok((
        FooterPrune::new(opened(buffer)?, filter, SESSION.clone(), next),
        invoked,
    ))
}

#[rstest]
#[case::provably_false(Some(gt(col("numbers"), lit(100u32))), false)]
#[case::possibly_true(Some(gt(col("numbers"), lit(3u32))), true)]
#[case::no_filter(None, true)]
fn prunes_by_file_statistics(
    #[case] filter: Option<Expression>,
    #[case] survives: bool,
) -> VortexResult<()> {
    let (mut stage, invoked) = prune(write_test_file(&columns())?, filter)?;
    assert_eq!(stage.state(), State::NeedsCompute);
    let output = stage.compute()?;
    if survives {
        assert!(
            matches!(&output, PlannerOutput::Planner(scope, _) if scope.rows == (0..8)),
            "{output:?}"
        );
    } else {
        assert!(matches!(output, PlannerOutput::Done), "{output:?}");
    }
    assert_eq!(invoked.load(Ordering::SeqCst), survives);
    assert_eq!(stage.state(), State::Done);
    Ok(())
}

#[test]
fn missing_statistics_retain_the_file() -> VortexResult<()> {
    let (mut stage, invoked) = prune(
        write_no_stats_test_file(&columns())?,
        Some(gt(col("numbers"), lit(100u32))),
    )?;
    assert!(matches!(stage.compute()?, PlannerOutput::Planner(..)));
    assert!(invoked.load(Ordering::SeqCst));
    Ok(())
}

#[test]
fn empty_file_finishes_without_a_child() -> VortexResult<()> {
    let empty = write_test_file(&[("numbers", buffer![0u32; 0].into_array())])?;
    let (mut stage, invoked) = prune(empty, None)?;
    assert!(matches!(stage.compute()?, PlannerOutput::Done));
    assert!(!invoked.load(Ordering::SeqCst));
    assert_eq!(stage.state(), State::Done);
    Ok(())
}

#[test]
fn missing_column_surfaces_the_bind_error() -> VortexResult<()> {
    let (mut stage, invoked) = prune(
        write_test_file(&columns())?,
        Some(gt(col("missing"), lit(1u32))),
    )?;
    let err = stage.compute().err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("missing")),
        "{err:?}"
    );
    assert!(!invoked.load(Ordering::SeqCst));
    Ok(())
}
