// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewrite identical rows with flat lists or ListLayout for selection-scan experiments.
//!
//! Usage: `rewrite_list_layout INPUT OUTPUT flat|list|list-elements [TARGET_BYTES]`.
//! Creates a new output file, verifies full Arrow equality through V1, and prints layout metadata.
//! `list-elements` applies TARGET_BYTES only to elements; other children keep the default strategy.
//! Compare this mode at different targets to isolate element sizing from outer-column partitioning.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutRef;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::table::TableStrategy;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

// ListLayout is registered for reading but is not in the default write edition.
static EXPERIMENT_EDITION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: EditionId::new("list-layout-experiment", 2026, 10, 0),
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.list")],
};

async fn read_v1(file: &VortexFile) -> Result<ArrayRef> {
    Ok(ArrayStreamAdapter::new(
        file.dtype().clone(),
        file.scan()?.prepare()?.execute_stream(None)?,
    )
    .read_all()
    .await?)
}

fn count_layouts(layout: &LayoutRef, counts: &mut BTreeMap<String, usize>) -> Result<()> {
    *counts.entry(layout.encoding_id().to_string()).or_default() += 1;
    for child in layout.children()? {
        count_layouts(&child, counts)?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        (3..=4).contains(&args.len()),
        "Expected INPUT OUTPUT flat|list|list-elements [TARGET_BYTES]"
    );
    ensure!(
        matches!(args[2].as_str(), "flat" | "list" | "list-elements"),
        "Unknown layout"
    );
    let target_bytes = args.get(3).map(|value| value.parse::<u64>()).transpose()?;
    ensure!(target_bytes != Some(0), "TARGET_BYTES must be positive");
    let session = VortexSession::default().with_tokio();
    session.register_edition(&EXPERIMENT_EDITION)?;
    if args[2] != "flat" {
        session.enable_edition(EXPERIMENT_EDITION.edition.id)?;
    }
    let input = session
        .open_options()
        .open_path(Path::new(&args[0]))
        .await?;
    let mut strategy = WriteStrategyBuilder::from_session(&session);
    if let Some(target_bytes) = target_bytes {
        strategy = strategy.with_data_block_target_bytes(Some(target_bytes));
    }
    let strategy = if args[2] == "list-elements" {
        let elements = strategy.build();
        let defaults = WriteStrategyBuilder::from_session(&session).build();
        Arc::new(
            TableStrategy::new(Arc::clone(&defaults), defaults).with_list_layout_factory(
                move |list| Arc::new(list.with_elements(Arc::clone(&elements))),
            ),
        ) as Arc<dyn LayoutStrategy>
    } else {
        strategy.build()
    };
    let stream = ArrayStreamAdapter::new(
        input.dtype().clone(),
        input.scan()?.prepare()?.execute_stream(None)?,
    );
    let mut output = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])
        .await?;
    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(&mut output, stream)
        .await?;
    drop(output);
    let rewritten = session
        .open_options()
        .open_path(Path::new(&args[1]))
        .await?;
    let mut counts = BTreeMap::new();
    count_layouts(rewritten.footer().layout(), &mut counts)?;
    ensure!(
        (counts.get("vortex.list").copied().unwrap_or(0) > 0) == (args[2] != "flat"),
        "Written layout does not match the requested representation"
    );
    ensure!(
        input.dtype() == rewritten.dtype(),
        "Rewritten schema differs"
    );
    let expected = read_v1(&input).await?;
    let actual = read_v1(&rewritten).await?;
    let mut ctx = session.create_execution_ctx();
    let arrow = session.arrow();
    let target = arrow.to_arrow_field("", input.dtype())?;
    let expected = arrow.execute_arrow(expected, Some(&target), &mut ctx)?;
    let actual = arrow.execute_arrow(actual, Some(&target), &mut ctx)?;
    ensure!(
        expected.as_ref() == actual.as_ref(),
        "Rewritten values differ"
    );
    println!(
        "{}",
        serde_json::json!({
            "input": args[0], "output": args[1], "representation": args[2],
            "target_bytes": target_bytes, "rows": summary.row_count(),
            "file_bytes": summary.size(), "correct": true, "layouts": counts,
            "layout_tree": rewritten.footer().layout().display_tree().to_string(),
        })
    );
    Ok(())
}
