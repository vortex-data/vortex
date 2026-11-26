// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::TryStreamExt;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::col;
use vortex_expr::gt_eq;
use vortex_expr::lit;
use vortex_file::DeserializeStep;
use vortex_file::Footer;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::filesystem::FileSystemRef;
use vortex_io::object_store::ObjectStoreFileSystem;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::FileSystemOpenable;
use crate::Manifest;
use crate::ManifestEntry;
use crate::Openable;
use crate::ReadAtOpenable;
use crate::TableScan;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

async fn data_file(rows: std::ops::Range<u64>) -> VortexResult<ByteBuffer> {
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(
            &mut buf,
            StructArray::from_fields(&[("x", PrimitiveArray::from_iter(rows).into_array())])?
                .into_array()
                .to_array_stream(),
        )
        .await?;
    Ok(buf.freeze())
}

async fn manifest(files: &[ByteBuffer]) -> VortexResult<Manifest> {
    let mut manifest = Manifest::default();
    for (i, bytes) in files.iter().enumerate() {
        let path = format!("data/part-{i}.vortex");
        let entry = ManifestEntry::read(&SESSION, path, Arc::new(bytes.clone())).await?;
        manifest.entries.push(entry);
    }
    Ok(manifest)
}

#[tokio::test]
async fn a_written_manifest_reads_back_the_same() -> VortexResult<()> {
    let files = [
        data_file(0..1000).await?,
        data_file(1000..1500).await?,
        data_file(1500..4000).await?,
    ];
    let manifest = manifest(&files).await?;
    assert_eq!(manifest.entries[1].row_count, 500);

    let mut written = ByteBufferMut::empty();
    manifest.write(&SESSION, &mut written).await?;
    let read = Manifest::read(&SESSION, Arc::new(written.freeze())).await?;
    assert_eq!(read, manifest);
    Ok(())
}

#[tokio::test]
async fn an_entry_records_the_exact_footer_region() -> VortexResult<()> {
    let bytes = data_file(0..2048).await?;
    let manifest = manifest(std::slice::from_ref(&bytes)).await?;
    let entry = &manifest.entries[0];

    assert_eq!(entry.byte_size, bytes.len() as u64);
    // The region from the recorded offset is enough to parse the footer: no more data is asked
    // for.
    let region = bytes.slice(usize::try_from(entry.footer_offset)?..);
    let step = Footer::deserializer(region, SESSION.clone())
        .with_size(entry.byte_size)
        .deserialize()?;
    let DeserializeStep::Done(footer) = step else {
        vortex_error::vortex_bail!("the region is not the whole footer");
    };
    assert_eq!(footer.row_count(), 2048);
    Ok(())
}

#[tokio::test]
async fn an_empty_manifest_reads_back_empty() -> VortexResult<()> {
    let mut written = ByteBufferMut::empty();
    Manifest::default().write(&SESSION, &mut written).await?;
    let read = Manifest::read(&SESSION, Arc::new(written.freeze())).await?;
    assert_eq!(read, Manifest::default());
    Ok(())
}

#[tokio::test]
async fn a_manifest_with_the_wrong_column_types_is_an_error() -> VortexResult<()> {
    // The manifest's column names, but signed sizes: reading must fail, not panic.
    let wrong = StructArray::from_fields(&[
        ("path", VarBinArray::from(vec!["a"]).into_array()),
        ("byte_size", PrimitiveArray::from_iter([1i64]).into_array()),
        ("row_count", PrimitiveArray::from_iter([1u64]).into_array()),
        (
            "footer_offset",
            PrimitiveArray::from_iter([0u64]).into_array(),
        ),
    ])?;
    let mut written = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut written, wrong.into_array().to_array_stream())
        .await?;
    assert!(
        Manifest::read(&SESSION, Arc::new(written.freeze()))
            .await
            .is_err()
    );
    Ok(())
}

/// Three files holding `x` = 0..1000, 1000..1500 and 1500..4000, and a way to open them.
async fn table() -> VortexResult<(
    Manifest,
    impl Fn(&ManifestEntry) -> VortexResult<Arc<dyn Openable>>,
)> {
    let files = [
        data_file(0..1000).await?,
        data_file(1000..1500).await?,
        data_file(1500..4000).await?,
    ];
    let manifest = manifest(&files).await?;
    let open = move |entry: &ManifestEntry| -> VortexResult<Arc<dyn Openable>> {
        let path = entry.path.as_str();
        let i: usize = path
            .trim_start_matches("data/part-")
            .trim_end_matches(".vortex")
            .parse()
            .map_err(|_| vortex_err!("no file at {path}"))?;
        Ok(Arc::new(ReadAtOpenable::new(
            path,
            Arc::new(files[i].clone()),
        )))
    };
    Ok((manifest, open))
}

async fn collect(scan: &TableScan, files: Vec<crate::PlannedFile>) -> VortexResult<ArrayRef> {
    let dtype = files[0].file.dtype().clone();
    let chunks: Vec<ArrayRef> = scan.run(files)?.try_collect().await?;
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

fn xs(rows: std::ops::Range<u64>) -> VortexResult<ArrayRef> {
    Ok(
        StructArray::from_fields(&[("x", PrimitiveArray::from_iter(rows).into_array())])?
            .into_array(),
    )
}

#[tokio::test]
async fn a_scan_returns_every_row_in_table_order() -> VortexResult<()> {
    let (manifest, open) = table().await?;
    let scan = TableScan::new(SESSION.clone());
    let files = scan.plan(&manifest, open).await?;
    assert_eq!(files.len(), 3);
    let actual = collect(&scan, files).await?;
    assert_arrays_eq!(actual, xs(0..4000)?, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[tokio::test]
async fn a_filter_prunes_files_on_their_statistics() -> VortexResult<()> {
    let (manifest, open) = table().await?;
    let scan = TableScan::new(SESSION.clone()).with_filter(gt_eq(col("x"), lit(1600u64)));
    let files = scan.plan(&manifest, open).await?;
    // Only the third file can hold x >= 1600; it starts at table row 1500.
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].row_offset, 1500);
    let actual = collect(&scan, files).await?;
    assert_arrays_eq!(actual, xs(1600..4000)?, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[tokio::test]
async fn a_scan_opens_files_on_a_file_system() -> VortexResult<()> {
    let files = [
        data_file(0..1000).await?,
        data_file(1000..1500).await?,
        data_file(1500..4000).await?,
    ];
    let manifest = manifest(&files).await?;
    let store = InMemory::new();
    for (entry, bytes) in manifest.entries.iter().zip(&files) {
        store
            .put(
                &ObjectPath::from(entry.path.as_str()),
                bytes.to_vec().into(),
            )
            .await?;
    }
    let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(
        Arc::new(store),
        SESSION.handle(),
    ));
    let scan = TableScan::new(SESSION.clone()).with_filter(gt_eq(col("x"), lit(1200u64)));
    let files = scan
        .plan(&manifest, |entry| {
            Ok(Arc::new(FileSystemOpenable::new(
                Arc::clone(&fs),
                entry.path.as_str(),
            )))
        })
        .await?;
    assert_eq!(
        files.iter().map(|file| file.row_offset).collect::<Vec<_>>(),
        vec![1000, 1500]
    );
    let actual = collect(&scan, files).await?;
    assert_arrays_eq!(actual, xs(1200..4000)?, &mut SESSION.create_execution_ctx());
    Ok(())
}
