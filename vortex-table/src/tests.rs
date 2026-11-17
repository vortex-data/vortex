// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::DeserializeStep;
use vortex_file::Footer;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::Manifest;
use crate::ManifestBuilder;
use crate::manifest::footer_hash;

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

async fn manifest(files: &[ByteBuffer], inline_footers: bool) -> VortexResult<Manifest> {
    let mut builder = ManifestBuilder::new(SESSION.clone()).with_inline_footers(inline_footers);
    for (i, bytes) in files.iter().enumerate() {
        builder
            .add(format!("data/part-{i}.vortex"), Arc::new(bytes.clone()))
            .await?;
    }
    Ok(builder.finish())
}

#[rstest]
#[case::inlined(true)]
#[case::not_inlined(false)]
#[tokio::test]
async fn a_written_manifest_reads_back_the_same(#[case] inline: bool) -> VortexResult<()> {
    let files = [
        data_file(0..1000).await?,
        data_file(1000..1500).await?,
        data_file(1500..4000).await?,
    ];
    let manifest = manifest(&files, inline).await?;
    assert_eq!(manifest.row_count(), 4000);
    assert_eq!(manifest.entries()[1].row_count, 500);
    for entry in manifest.entries() {
        assert_eq!(entry.footer.is_some(), inline);
    }

    let mut written = ByteBufferMut::empty();
    manifest.write(&SESSION, &mut written).await?;
    let read = Manifest::read(&SESSION, Arc::new(written.freeze())).await?;
    assert_eq!(read, manifest);
    Ok(())
}

#[tokio::test]
async fn an_entry_records_the_exact_footer_region() -> VortexResult<()> {
    let bytes = data_file(0..2048).await?;
    let manifest = manifest(std::slice::from_ref(&bytes), true).await?;
    let entry = &manifest.entries()[0];

    assert_eq!(entry.byte_size, bytes.len() as u64);
    assert_eq!(entry.footer_offset + entry.footer_len, entry.byte_size);
    let tail = bytes.slice(usize::try_from(entry.footer_offset)?..bytes.len());
    let inlined = entry
        .footer
        .clone()
        .ok_or_else(|| vortex_error::vortex_err!("inlined"))?;
    assert_eq!(inlined, tail);
    assert_eq!(entry.footer_hash, footer_hash(&tail));
    // The region alone is enough to parse the footer: no more data is asked for.
    let step = Footer::deserializer(inlined, SESSION.clone())
        .with_size(entry.byte_size)
        .deserialize()?;
    let DeserializeStep::Done(footer) = step else {
        vortex_error::vortex_bail!("the inlined region is not the whole footer");
    };
    assert_eq!(footer.row_count(), 2048);
    Ok(())
}

#[tokio::test]
async fn rewriting_a_file_changes_its_footer_hash() -> VortexResult<()> {
    let before = manifest(&[data_file(0..1000).await?], false).await?;
    let after = manifest(&[data_file(5000..6000).await?], false).await?;
    assert_ne!(
        before.entries()[0].footer_hash,
        after.entries()[0].footer_hash
    );
    Ok(())
}
