// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::DeserializeStep;
use vortex_file::Footer;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::Manifest;
use crate::ManifestEntry;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| VortexSession::default().with_tokio());

async fn data_file(rows: std::ops::Range<u64>) -> VortexResult<ByteBuffer> {
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(
            &mut buf,
            PrimitiveArray::from_iter(rows)
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
