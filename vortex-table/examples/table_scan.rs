// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writes a table of ten Vortex files and a manifest, then scans it with a filter: each file opens
//! with one read of its footer, found at the offset the manifest records, the files' statistics
//! rule out the ones that cannot match, and each kept file runs its ordinary scan.
//!
//! ```sh
//! cargo run -p vortex-table --example table_scan
//! ```

use std::sync::Arc;

use futures::TryStreamExt;
use vortex::VortexSessionDefault;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_error::VortexResult;
use vortex_expr::col;
use vortex_expr::gt_eq;
use vortex_expr::lit;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::std_file::FileReadAt;
use vortex_session::VortexSession;
use vortex_table::Manifest;
use vortex_table::ManifestEntry;
use vortex_table::TableScan;

const FILES: u64 = 10;
const ROWS_PER_FILE: u64 = 100_000;

#[tokio::main]
async fn main() -> VortexResult<()> {
    let session = VortexSession::default().with_tokio();
    let dir = tempfile::tempdir()?;
    let open = |path: &str| -> VortexResult<Arc<dyn VortexReadAt>> {
        Ok(Arc::new(FileReadAt::open(
            dir.path().join(path),
            session.handle(),
        )?))
    };

    // Write the data files, each holding the next range of `x`, and list them in a manifest.
    let mut manifest = Manifest::default();
    for i in 0..FILES {
        let path = format!("part-{i}.vortex");
        let rows = i * ROWS_PER_FILE..(i + 1) * ROWS_PER_FILE;
        let data =
            StructArray::from_fields(&[("x", PrimitiveArray::from_iter(rows).into_array())])?;
        session
            .write_options()
            .write(
                tokio::fs::File::create(dir.path().join(&path)).await?,
                data.into_array().to_array_stream(),
            )
            .await?;
        manifest
            .entries
            .push(ManifestEntry::read(&session, &path, open(&path)?).await?);
    }
    let manifest_path = dir.path().join("manifest.vortex");
    manifest
        .write(&session, tokio::fs::File::create(&manifest_path).await?)
        .await?;

    // Read the manifest back and scan the table for x >= 750,000.
    let manifest = Manifest::read(
        &session,
        Arc::new(FileReadAt::open(&manifest_path, session.handle())?),
    )
    .await?;
    let scan = TableScan::new(session.clone()).with_filter(gt_eq(col("x"), lit(750_000u64)));
    let files = scan.plan(&manifest, &open).await?;
    println!("kept {} of {} files", files.len(), manifest.entries.len());
    let rows: usize = scan
        .run(files)?
        .try_fold(0, |rows, chunk| async move { Ok(rows + chunk.len()) })
        .await?;
    println!("scanned {rows} rows");
    Ok(())
}
