// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use flatbuffers::FlatBufferBuilder;
use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use tempfile::tempdir;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::flatbuffers::ReadFlatBuffer;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSession;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::envelope::EnvelopeEncryption;
use super::envelope::InMemoryKeyMaterialStore;
use super::envelope::KeyMaterialStore;
use super::envelope::KmsClient;
use super::*;
use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::Footer;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
use crate::WriteStrategyBuilder;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptSegment;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

const FOOTER_KEY_METADATA: &[u8] = b"kms://keys/footer";
const COLUMN_KEY_METADATA: &[u8] = b"kms://keys/pii";
const SECRET: &str = "a-secret-string-value";
const SECRET_COLUMN: &str = "secret_column_name";
const VISIBLE: &str = "a-visible-string-value";
const AAD: &[u8] = b"s3://bucket/table/part-0.vortex";

fn footer_key() -> KeyConfig {
    KeyConfig::new(EncryptionKey::from([42; 32]), FOOTER_KEY_METADATA)
}

fn column_key() -> KeyConfig {
    KeyConfig::new(EncryptionKey::from([7; 32]), COLUMN_KEY_METADATA)
}

/// Resolves the footer key, and the column key only if `with_column_key`.
fn keys(with_column_key: bool) -> Arc<dyn KeyRetriever> {
    Arc::new(move |key_metadata: &[u8]| match key_metadata {
        FOOTER_KEY_METADATA => Ok(footer_key().key().clone()),
        COLUMN_KEY_METADATA if with_column_key => Ok(column_key().key().clone()),
        _ => vortex_bail!("access denied to key {key_metadata:?}"),
    })
}

fn decryption() -> FileDecryption {
    FileDecryption::new(keys(true))
}

fn array() -> VortexResult<ArrayRef> {
    let secret = VarBinArray::from(vec![SECRET, "foo", SECRET, "bar"]).into_array();
    let visible = VarBinArray::from(vec![VISIBLE, "baz", VISIBLE, "qux"]).into_array();
    let numbers = buffer![1u32, 2, 3, 4].into_array();
    Ok(StructArray::from_fields(&[
        (SECRET_COLUMN, secret),
        ("visible", visible),
        ("numbers", numbers),
    ])?
    .into_array())
}

/// Encrypt the secret column with its own key.
fn with_column_key(encryption: FileEncryption) -> FileEncryption {
    encryption.with_field_key(FieldPath::from_name(SECRET_COLUMN), "pii", column_key())
}

async fn write(encryption: Option<FileEncryption>) -> VortexResult<ByteBuffer> {
    let mut buf = ByteBufferMut::empty();
    let mut options = SESSION
        .write_options()
        .with_metadata_segment("meta", ByteBuffer::copy_from(SECRET.as_bytes()));
    if let Some(encryption) = encryption {
        options = options.with_encryption(encryption);
    }
    options.write(&mut buf, array()?.to_array_stream()).await?;
    Ok(buf.freeze())
}

fn open(bytes: ByteBuffer, decryption: Option<FileDecryption>) -> VortexResult<VortexFile> {
    let mut options = SESSION.open_options().include_metadata();
    if let Some(decryption) = decryption {
        options = options.with_decryption(decryption);
    }
    options.open_buffer(bytes)
}

async fn read_all(file: &VortexFile) -> VortexResult<ArrayRef> {
    let chunks: Vec<ArrayRef> = file.scan()?.into_array_stream()?.try_collect().await?;
    Ok(ChunkedArray::try_new(chunks, file.dtype().clone())?.into_array())
}

async fn read_field(file: &VortexFile, field: &str) -> VortexResult<ArrayRef> {
    let projection = select([field], root())
        .bind(file.dtype())?
        .optimize_recursive()?;
    file.scan()?
        .with_projection(projection)
        .into_array_stream()?
        .read_all()
        .await
}

/// Deserialize a standalone footer, retrieving its keys when asked to.
async fn deserialize_footer(bytes: &[u8], decryption: FileDecryption) -> VortexResult<Footer> {
    let mut deserializer = Footer::deserializer(ByteBuffer::copy_from(bytes), SESSION.clone())
        .with_size(bytes.len() as u64)
        .with_decryption(decryption);
    loop {
        match deserializer.deserialize()? {
            DeserializeStep::NeedKeys => deserializer.resolve_keys().await,
            DeserializeStep::Done(footer) => return Ok(footer),
            step => vortex_bail!("standalone footer needs no {step:?}"),
        }
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn expect_err<T>(result: VortexResult<T>, expected: &str) {
    match result {
        Ok(_) => panic!("expected an error containing {expected:?}"),
        Err(err) => assert!(err.to_string().contains(expected), "{err}"),
    }
}

async fn assert_reads_all(file: &VortexFile) -> VortexResult<()> {
    assert_arrays_eq!(
        read_all(file).await?,
        array()?,
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn key_lengths() -> VortexResult<()> {
    for len in KEY_LENGTHS {
        assert_eq!(EncryptionKey::generate(len)?.as_bytes().len(), len);
    }
    for len in [0, 15, 31, 33] {
        assert!(EncryptionKey::try_from_slice(&vec![0; len]).is_err());
    }
    Ok(())
}

#[test]
fn spec_flatbuffer_roundtrip() -> VortexResult<()> {
    let spec = FileEncryption::new(KeyConfig::new(
        EncryptionKey::from([7; 32]),
        b"k".as_slice(),
    ))
    .with_aad_prefix(b"prefix".as_slice())
    .file_cipher(&KeyConfig::new(
        EncryptionKey::from([7; 16]),
        b"k".as_slice(),
    ))?
    .spec()
    .clone();
    let mut fbb = FlatBufferBuilder::new();
    let offset = spec.write_flatbuffer(&mut fbb);
    fbb.finish_minimal(offset);
    let read = EncryptionSpec::read_flatbuffer(&flatbuffers::root::<
        crate::flatbuffers::footer::EncryptionSpec,
    >(fbb.finished_data())?)?;
    assert_eq!(read, spec);
    assert!(read.requires_aad_prefix());
    Ok(())
}

#[tokio::test]
async fn roundtrip() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;

    // Neither the data, nor the schema, nor user metadata are stored in plaintext.
    for value in [SECRET, SECRET_COLUMN, VISIBLE] {
        assert!(!contains(&bytes, value.as_bytes()), "{value}");
    }

    // The postscript stores the footer key's spec once, however many segments it protects.
    assert_eq!(
        bytes
            .windows(FOOTER_KEY_METADATA.len())
            .filter(|window| *window == FOOTER_KEY_METADATA)
            .count(),
        1
    );

    let file = open(bytes, Some(decryption()))?;
    assert!(!file.footer().encryption_specs().is_empty());
    assert_eq!(
        file.metadata_segment("meta").map(ByteBuffer::as_slice),
        Some(SECRET.as_bytes())
    );
    assert_reads_all(&file).await
}

/// A segment cache recording what it holds.
#[derive(Default)]
struct RecordingCache {
    segments: Mutex<HashMap<SegmentId, ByteBuffer>>,
}

#[async_trait::async_trait]
impl SegmentCache for RecordingCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.segments.lock().get(&id).cloned())
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segments.lock().insert(id, buffer);
        Ok(())
    }
}

async fn open_with_cache(
    bytes: ByteBuffer,
    decryption: FileDecryption,
) -> VortexResult<(VortexFile, Arc<RecordingCache>)> {
    let cache = Arc::new(RecordingCache::default());
    let file = SESSION
        .open_options()
        .with_decryption(decryption)
        .with_segment_cache(Arc::clone(&cache) as _)
        .open_read(bytes)
        .await?;
    Ok((file, cache))
}

/// By default the segment cache holds ciphertext; with a plaintext cache it holds plaintext.
#[tokio::test]
async fn plaintext_cache() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    let cached_data_segment = |file: &VortexFile, cache: &RecordingCache| {
        let segment = file.footer().segment_map()[0];
        let ciphertext = bytes.slice(
            usize::try_from(segment.offset).vortex_expect("offset")
                ..usize::try_from(segment.offset + u64::from(segment.length)).vortex_expect("end"),
        );
        let cached = cache
            .segments
            .lock()
            .get(&SegmentId::from(0))
            .cloned()
            .vortex_expect("segment 0 cached");
        (ciphertext, cached)
    };

    let (file, cache) = open_with_cache(bytes.clone(), decryption()).await?;
    assert_reads_all(&file).await?;
    let (ciphertext, cached) = cached_data_segment(&file, &cache);
    assert_eq!(cached.as_slice(), ciphertext.as_slice());

    let (file, cache) = open_with_cache(bytes.clone(), decryption().with_plaintext_cache()).await?;
    assert_reads_all(&file).await?;
    let (ciphertext, cached) = cached_data_segment(&file, &cache);
    assert_eq!(cached.len() + NONCE_TAG_LEN, ciphertext.len());
    assert_ne!(cached.as_slice(), &ciphertext[..cached.len()]);
    // A second scan is served from the plaintext cache.
    assert_reads_all(&file).await?;

    // Without the column key, its segments stay out of the plaintext cache and the other
    // fields remain readable.
    let (file, _) = open_with_cache(
        bytes,
        FileDecryption::new(keys(false)).with_plaintext_cache(),
    )
    .await?;
    read_field(&file, "numbers").await?;
    expect_err(read_field(&file, SECRET_COLUMN).await, "access denied");
    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn roundtrip_path() -> VortexResult<()> {
    let dir = tempdir()?;
    let path = dir.path().join("encrypted.vortex");
    fs::write(
        &path,
        write(Some(FileEncryption::new(footer_key())))
            .await?
            .as_slice(),
    )?;

    let file = SESSION
        .open_options()
        .with_decryption(decryption())
        .include_metadata()
        .open_path(&path)
        .await?;
    assert_eq!(
        file.metadata_segment("meta").map(ByteBuffer::as_slice),
        Some(SECRET.as_bytes())
    );
    assert_reads_all(&file).await
}

#[rstest]
#[case::aes_128(16)]
#[case::aes_192(24)]
#[case::aes_256(32)]
#[tokio::test]
async fn key_lengths_roundtrip(#[case] key_len: usize) -> VortexResult<()> {
    let key = EncryptionKey::generate(key_len)?;
    let bytes = write(Some(FileEncryption::new(KeyConfig::new(
        key.clone(),
        b"key".as_slice(),
    ))))
    .await?;
    assert!(!contains(&bytes, SECRET.as_bytes()));

    let file = open(bytes, Some(FileDecryption::new(Arc::new(key))))?;
    assert!(
        file.footer()
            .encryption_specs()
            .iter()
            .skip(1)
            .all(|spec| spec.algorithm() == EncryptionAlgorithm::AesGcm)
    );
    assert_reads_all(&file).await
}

#[tokio::test]
async fn none_algorithm_fails() -> VortexResult<()> {
    let result = write(Some(
        FileEncryption::new(footer_key()).with_algorithm(EncryptionAlgorithm::None),
    ))
    .await;
    expect_err(result, "cannot be used to encrypt");
    Ok(())
}

#[tokio::test]
async fn requires_decryption() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    expect_err(open(bytes, None), "no FileDecryption");
    Ok(())
}

#[tokio::test]
async fn rejects_wrong_key() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let wrong = FileDecryption::new(Arc::new(EncryptionKey::from([0; 32])));
    expect_err(open(bytes, Some(wrong)), "Failed to decrypt");
    Ok(())
}

#[tokio::test]
async fn rejects_tampered_data_segment() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let segment = open(bytes.clone(), Some(decryption()))?
        .footer()
        .segment_map()[0];

    let mut tampered = ByteBufferMut::copy_from(&bytes);
    tampered[usize::try_from(segment.offset)?] ^= 1;
    let file = open(tampered.freeze(), Some(decryption()))?;
    expect_err(read_all(&file).await, "Failed to decrypt");
    Ok(())
}

#[tokio::test]
async fn plaintext_files_are_rejected_unless_allowed() -> VortexResult<()> {
    let bytes = write(None).await?;
    expect_err(open(bytes.clone(), Some(decryption())), "not encrypted");

    let file = open(bytes, Some(decryption().with_plaintext_files_allowed()))?;
    assert_reads_all(&file).await
}

#[tokio::test]
async fn standalone_footer() -> VortexResult<()> {
    let encryption = FileEncryption::new(footer_key());
    let bytes = write(Some(encryption.clone())).await?;
    let footer = open(bytes.clone(), Some(decryption()))?.footer().clone();

    expect_err(
        footer.clone().into_serializer().serialize(),
        "with_encryption",
    );

    // A standalone footer, encrypted under the same key, opens the file without reading its
    // footer.
    let footer_bytes = footer
        .into_serializer()
        .with_encryption(&encryption)?
        .serialize()?
        .iter()
        .flat_map(|buffer| buffer.as_slice().iter().copied())
        .collect::<Vec<u8>>();
    let footer = deserialize_footer(&footer_bytes, decryption()).await?;
    let file = SESSION
        .open_options()
        .with_decryption(decryption())
        .with_footer(footer)
        .open_buffer(bytes)?;
    assert_reads_all(&file).await
}

/// A standalone footer keeps the statistics of a column-keyed field under that key.
#[tokio::test]
async fn standalone_footer_with_field_key() -> VortexResult<()> {
    let encryption = with_column_key(FileEncryption::new(footer_key()));
    let bytes = write(Some(encryption.clone())).await?;
    let footer = open(bytes, Some(decryption()))?.footer().clone();

    let footer_bytes = footer
        .into_serializer()
        .with_encryption(&encryption)?
        .serialize()?
        .iter()
        .flat_map(|buffer| buffer.as_slice().iter().copied())
        .collect::<Vec<u8>>();
    for (with_column_key, expect_secret_stats) in [(true, true), (false, false)] {
        let footer =
            deserialize_footer(&footer_bytes, FileDecryption::new(keys(with_column_key))).await?;
        assert_eq!(footer.statistics_encryption().as_ref(), [(0, 2)]);
        let stats = footer.statistics().vortex_expect("file statistics");
        assert_eq!(!stats.aggregates()[0].is_empty(), expect_secret_stats);
        assert!(!stats.aggregates()[2].is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn field_key() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    for value in [SECRET, SECRET_COLUMN, VISIBLE] {
        assert!(!contains(&bytes, value.as_bytes()), "{value}");
    }

    let file = open(bytes, Some(decryption()))?;
    let footer = file.footer();
    let key_metadata: Vec<&[u8]> = footer
        .encryption_specs()
        .iter()
        .map(|spec| spec.key_metadata())
        .collect();
    assert_eq!(
        key_metadata,
        [b"".as_slice(), FOOTER_KEY_METADATA, COLUMN_KEY_METADATA]
    );

    // Every segment of the secret column, and only those, uses the column key.
    // Slot 0 holds the struct validity, followed by the fields.
    let secret_field = footer.layout().slot(1)?.vortex_expect("secret column");
    let mut secret_segments = Vec::new();
    for layout in secret_field.depth_first_traversal() {
        secret_segments.extend(layout?.segment_ids().iter().map(|id| **id as usize));
    }
    assert!(!secret_segments.is_empty());
    for (index, segment) in footer.segment_map().iter().enumerate() {
        let expected = if secret_segments.contains(&index) {
            2
        } else {
            1
        };
        assert_eq!(segment.encryption, expected, "segment {index}");
    }

    // The secret column's statistics are encrypted with its key, in a segment of their own.
    assert_eq!(footer.statistics_encryption().as_ref(), [(0, 2)]);
    let stats = footer.statistics().vortex_expect("file statistics");
    assert!(stats.aggregates().iter().all(|stats| !stats.is_empty()));

    assert_reads_all(&file).await
}

#[tokio::test]
async fn column_key_unavailable() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    let file = open(bytes, Some(FileDecryption::new(keys(false))))?;

    // Without the column key, the secret column's statistics are unavailable too.
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates()[0].is_empty());
    assert!(!stats.aggregates()[2].is_empty());

    // Columns encrypted with the footer key remain readable.
    let expected = StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3, 4].into_array())])?;
    assert_arrays_eq!(
        read_field(&file, "numbers").await?,
        expected,
        &mut SESSION.create_execution_ctx()
    );
    expect_err(read_field(&file, SECRET_COLUMN).await, "access denied");
    Ok(())
}

#[tokio::test]
async fn unregistered_key_name_fails() -> VortexResult<()> {
    let strategy = WriteStrategyBuilder::from_session(&SESSION)
        .with_field_encryption(FieldPath::from_name(SECRET_COLUMN), "missing")
        .build();
    let result = SESSION
        .write_options()
        .with_strategy(strategy)
        .with_encryption(FileEncryption::new(footer_key()))
        .write(ByteBufferMut::empty(), array()?.to_array_stream())
        .await;
    expect_err(result, "No encryption key named missing");
    Ok(())
}

#[tokio::test]
async fn field_key_requires_default_strategy() -> VortexResult<()> {
    let result = SESSION
        .write_options()
        .with_strategy(WriteStrategyBuilder::from_session(&SESSION).build())
        .with_encryption(with_column_key(FileEncryption::new(footer_key())))
        .write(ByteBufferMut::empty(), array()?.to_array_stream())
        .await;
    expect_err(result, "default strategy only");
    Ok(())
}

/// Fields without their own key are stored in plaintext, while the footer stays encrypted.
#[tokio::test]
async fn plaintext_fields() -> VortexResult<()> {
    let encryption = with_column_key(FileEncryption::new(footer_key()).with_plaintext_fields());
    let bytes = write(Some(encryption)).await?;
    assert!(contains(&bytes, VISIBLE.as_bytes()));
    assert!(!contains(&bytes, SECRET.as_bytes()));
    assert!(!contains(&bytes, SECRET_COLUMN.as_bytes()));

    let file = open(bytes, Some(decryption()))?;
    let footer = file.footer();
    // Slot 2 holds the visible column.
    let visible_field = footer.layout().slot(2)?.vortex_expect("visible column");
    for layout in visible_field.depth_first_traversal() {
        for id in layout?.segment_ids() {
            assert_eq!(footer.segment_map()[*id as usize].encryption, 0);
        }
    }
    assert_reads_all(&file).await
}

/// With a plaintext footer, readers without keys can read the plaintext fields, and readers with
/// the footer key verify its signature.
#[tokio::test]
async fn plaintext_footer() -> VortexResult<()> {
    let encryption = with_column_key(
        FileEncryption::new(footer_key())
            .with_plaintext_footer()
            .with_plaintext_fields(),
    );
    let bytes = write(Some(encryption)).await?;
    // The schema is visible, but not the data or user metadata of the encrypted column.
    assert!(contains(&bytes, SECRET_COLUMN.as_bytes()));
    assert!(contains(&bytes, VISIBLE.as_bytes()));
    assert!(!contains(&bytes, SECRET.as_bytes()));

    // A reader without encryption support reads the plaintext fields.
    let file = SESSION.open_options().open_buffer(bytes.clone())?;
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates()[0].is_empty());
    assert!(!stats.aggregates()[2].is_empty());
    let expected = StructArray::from_fields(&[("numbers", buffer![1u32, 2, 3, 4].into_array())])?;
    assert_arrays_eq!(
        read_field(&file, "numbers").await?,
        expected,
        &mut SESSION.create_execution_ctx()
    );
    expect_err(read_field(&file, SECRET_COLUMN).await, "no FileDecryption");

    let file = open(bytes, Some(decryption()))?;
    assert_eq!(
        file.metadata_segment("meta").map(ByteBuffer::as_slice),
        Some(SECRET.as_bytes())
    );
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates().iter().all(|stats| !stats.is_empty()));
    assert_reads_all(&file).await
}

/// With a plaintext footer and every field encrypted with the footer key, the file statistics
/// are encrypted with the footer key.
#[tokio::test]
async fn plaintext_footer_with_encrypted_fields() -> VortexResult<()> {
    let bytes = write(Some(
        FileEncryption::new(footer_key()).with_plaintext_footer(),
    ))
    .await?;
    assert!(contains(&bytes, SECRET_COLUMN.as_bytes()));
    assert!(!contains(&bytes, VISIBLE.as_bytes()));

    let file = SESSION.open_options().open_buffer(bytes.clone())?;
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates().iter().all(|stats| stats.is_empty()));
    assert_eq!(
        file.footer().statistics_encryption().as_ref(),
        [(0, 1), (1, 1), (2, 1)]
    );
    expect_err(read_field(&file, "numbers").await, "no FileDecryption");

    let file = open(bytes, Some(decryption()))?;
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates().iter().all(|stats| !stats.is_empty()));
    assert_reads_all(&file).await
}

/// A custom strategy that encrypts a field with a named key gets the same statistics protection,
/// since the keys are taken from the segments each field wrote.
#[tokio::test]
async fn custom_strategy_statistics_encryption() -> VortexResult<()> {
    let strategy = WriteStrategyBuilder::from_session(&SESSION)
        .with_field_encryption(FieldPath::from_name(SECRET_COLUMN), "pii")
        .build();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .with_encryption(FileEncryption::new(footer_key()).with_named_key("pii", column_key()))
        .write(&mut buf, array()?.to_array_stream())
        .await?;
    let bytes = buf.freeze();

    let file = open(bytes.clone(), Some(FileDecryption::new(keys(false))))?;
    assert_eq!(file.footer().statistics_encryption().as_ref(), [(0, 2)]);
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates()[0].is_empty());
    assert!(!stats.aggregates()[1].is_empty());

    let file = open(bytes, Some(decryption()))?;
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates().iter().all(|stats| !stats.is_empty()));
    assert_reads_all(&file).await
}

#[tokio::test]
async fn plaintext_footer_signature() -> VortexResult<()> {
    let bytes = write(Some(
        FileEncryption::new(footer_key())
            .with_plaintext_footer()
            .with_plaintext_fields(),
    ))
    .await?;

    // A wrong footer key fails verification, unless verification is disabled.
    let wrong_keys: Arc<dyn KeyRetriever> = Arc::new(EncryptionKey::from([0; 32]));
    let wrong = FileDecryption::new(wrong_keys);
    expect_err(
        open(bytes.clone(), Some(wrong.clone())),
        "verify the signature",
    );
    let file = SESSION
        .open_options()
        .with_decryption(wrong.without_footer_verification())
        .open_buffer(bytes.clone())?;
    read_field(&file, "numbers").await?;

    // A modified schema fails verification.
    let offset = bytes
        .windows(SECRET_COLUMN.len())
        .position(|window| window == SECRET_COLUMN.as_bytes())
        .vortex_expect("plaintext schema");
    let mut tampered = ByteBufferMut::copy_from(&bytes);
    tampered[offset] ^= 1;
    expect_err(
        open(tampered.freeze(), Some(decryption())),
        "verify the signature",
    );
    Ok(())
}

#[tokio::test]
async fn aad_prefix() -> VortexResult<()> {
    let encryption = with_column_key(FileEncryption::new(footer_key()).with_aad_prefix(AAD));
    let bytes = write(Some(encryption)).await?;
    assert!(!contains(&bytes, AAD));

    let file = open(bytes.clone(), Some(decryption().with_aad_prefix(AAD)))?;
    assert!(
        file.footer()
            .encryption_specs()
            .iter()
            .skip(1)
            .all(|spec| spec.requires_aad_prefix())
    );
    assert_reads_all(&file).await?;

    expect_err(
        open(bytes.clone(), Some(decryption())),
        "requires an AAD prefix",
    );
    expect_err(
        open(
            bytes,
            Some(decryption().with_aad_prefix(b"s3://bucket/table/other.vortex".as_slice())),
        ),
        "Failed to decrypt",
    );
    Ok(())
}

#[tokio::test]
async fn aad_prefix_for_file_without_one_fails() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    expect_err(
        open(bytes, Some(decryption().with_aad_prefix(AAD))),
        "not bound to one",
    );
    Ok(())
}

/// A KMS with versioned master keys, counting its calls. Wrapped keys are the master key version
/// followed by the AES-GCM encrypted key.
#[derive(Default)]
struct TestKms {
    master_keys: Mutex<HashMap<String, Vec<Option<EncryptionKey>>>>,
    wraps: AtomicUsize,
    unwraps: AtomicUsize,
}

impl TestKms {
    fn new(master_key_ids: &[&str]) -> Self {
        let kms = Self::default();
        for id in master_key_ids {
            kms.rotate(id);
        }
        kms
    }

    /// Add a new version of the master key.
    fn rotate(&self, master_key_id: &str) {
        self.master_keys
            .lock()
            .entry(master_key_id.to_string())
            .or_default()
            .push(Some(
                EncryptionKey::generate(32).vortex_expect("random key"),
            ));
    }

    /// Destroy every version of the master key but the current one.
    fn retire_old_versions(&self, master_key_id: &str) {
        let mut master_keys = self.master_keys.lock();
        let versions = master_keys
            .get_mut(master_key_id)
            .vortex_expect("master key");
        let current = versions.len() - 1;
        versions[..current].fill(None);
    }
}

#[async_trait::async_trait]
impl KmsClient for TestKms {
    async fn wrap_key(&self, key: &[u8], master_key_id: &str) -> VortexResult<Vec<u8>> {
        self.wraps.fetch_add(1, Ordering::Relaxed);
        let master_keys = self.master_keys.lock();
        let versions = master_keys
            .get(master_key_id)
            .ok_or_else(|| vortex_err!("unknown master key {master_key_id}"))?;
        let version = versions.len() - 1;
        let master = versions[version].as_ref().vortex_expect("current version");
        let mut wrapped = key.to_vec();
        let nonce_tag = Aead::new(master)?.seal(master_key_id.as_bytes(), &mut wrapped)?;
        wrapped.extend_from_slice(&nonce_tag);
        wrapped.insert(0, u8::try_from(version)?);
        Ok(wrapped)
    }

    async fn unwrap_key(&self, wrapped_key: &[u8], master_key_id: &str) -> VortexResult<Vec<u8>> {
        self.unwraps.fetch_add(1, Ordering::Relaxed);
        let master_keys = self.master_keys.lock();
        let (version, wrapped) = wrapped_key
            .split_first()
            .ok_or_else(|| vortex_err!("empty wrapped key"))?;
        let master = master_keys
            .get(master_key_id)
            .and_then(|versions| versions.get(usize::from(*version)))
            .and_then(Option::as_ref)
            .ok_or_else(|| vortex_err!("master key {master_key_id} v{version} unavailable"))?;
        let len = wrapped
            .len()
            .checked_sub(NONCE_TAG_LEN)
            .ok_or_else(|| vortex_err!("wrapped key too short"))?;
        let mut key = wrapped[..len].to_vec();
        Aead::new(master)?
            .open(master_key_id.as_bytes(), &mut key, &wrapped[len..])
            .map_err(|()| vortex_err!("failed to unwrap key"))?;
        Ok(key)
    }
}

async fn write_envelope(envelope: &EnvelopeEncryption) -> VortexResult<ByteBuffer> {
    let encryption = FileEncryption::new(envelope.key_config("footer-mek").await?).with_field_key(
        FieldPath::from_name(SECRET_COLUMN),
        "pii",
        envelope.key_config("pii-mek").await?,
    );
    write(Some(encryption)).await
}

#[tokio::test]
async fn envelope_double_wrapping() -> VortexResult<()> {
    let kms = Arc::new(TestKms::new(&["footer-mek", "pii-mek"]));
    let writer = EnvelopeEncryption::new(Arc::clone(&kms) as _);
    let first = write_envelope(&writer).await?;
    let second = write_envelope(&writer).await?;
    // One key encryption key is wrapped per master key, however many files are written.
    assert_eq!(kms.wraps.load(Ordering::Relaxed), 2);

    let reader = Arc::new(EnvelopeEncryption::new(Arc::clone(&kms) as _));
    for bytes in [first, second] {
        let file = open(bytes, Some(FileDecryption::new(Arc::clone(&reader) as _)))?;
        assert_reads_all(&file).await?;
    }
    // Key encryption keys are cached by id when reading.
    assert_eq!(kms.unwraps.load(Ordering::Relaxed), 2);
    Ok(())
}

#[tokio::test]
async fn envelope_single_wrapping() -> VortexResult<()> {
    let kms = Arc::new(TestKms::new(&["footer-mek", "pii-mek"]));
    let envelope = Arc::new(
        EnvelopeEncryption::new(Arc::clone(&kms) as _)
            .with_single_wrapping()
            .with_data_key_length(32)?,
    );
    assert_eq!(
        envelope
            .key_config("footer-mek")
            .await?
            .key()
            .as_bytes()
            .len(),
        32
    );
    let bytes = write_envelope(&envelope).await?;
    assert_eq!(kms.wraps.load(Ordering::Relaxed), 3);

    let file = open(bytes, Some(FileDecryption::new(envelope)))?;
    assert_reads_all(&file).await
}

#[tokio::test]
async fn envelope_master_key_rotation() -> VortexResult<()> {
    let kms = Arc::new(TestKms::new(&["footer-mek", "pii-mek"]));
    let store = Arc::new(InMemoryKeyMaterialStore::default());
    let envelope = Arc::new(
        EnvelopeEncryption::new(Arc::clone(&kms) as _)
            .with_external_key_material(Arc::clone(&store) as _),
    );
    let bytes = write_envelope(&envelope).await?;
    assert_eq!(store.ids().await?.len(), 2);

    // After rotating the master keys and retiring their old versions, a reader without cached
    // keys can still read the file.
    for id in ["footer-mek", "pii-mek"] {
        kms.rotate(id);
    }
    let report = envelope.rotate_master_keys().await?;
    assert_eq!(report.rewrapped, 2);
    assert!(report.failed.is_empty());
    for id in ["footer-mek", "pii-mek"] {
        kms.retire_old_versions(id);
    }
    let reader = EnvelopeEncryption::new(Arc::clone(&kms) as _)
        .with_external_key_material(Arc::clone(&store) as _);
    let file = open(bytes.clone(), Some(FileDecryption::new(Arc::new(reader))))?;
    assert_reads_all(&file).await?;

    // The file only references the key material, so a reader needs the store.
    let without_store = EnvelopeEncryption::new(Arc::clone(&kms) as _);
    expect_err(
        open(bytes, Some(FileDecryption::new(Arc::new(without_store)))),
        "no key material store",
    );
    Ok(())
}

#[tokio::test]
async fn envelope_rotation_requires_external_key_material() {
    let envelope = EnvelopeEncryption::new(Arc::new(TestKms::new(&["mek"])));
    expect_err(envelope.rotate_master_keys().await, "external key material");
}

/// Expired KEKs are replaced, so that new files pick up master key rotations.
#[tokio::test]
async fn envelope_kek_lifetime() -> VortexResult<()> {
    let kms = Arc::new(TestKms::new(&["mek"]));
    let envelope = EnvelopeEncryption::new(Arc::clone(&kms) as _).with_kek_lifetime(Duration::ZERO);
    envelope.key_config("mek").await?;
    envelope.key_config("mek").await?;
    assert_eq!(kms.wraps.load(Ordering::Relaxed), 2);
    Ok(())
}

/// Key material that fails to rewrap is reported, and the rest is still rewrapped.
#[tokio::test]
async fn envelope_rotation_reports_failures() -> VortexResult<()> {
    let kms = Arc::new(TestKms::new(&["mek"]));
    let store = Arc::new(InMemoryKeyMaterialStore::default());
    let envelope = EnvelopeEncryption::new(Arc::clone(&kms) as _)
        .with_external_key_material(Arc::clone(&store) as _);
    envelope.key_config("mek").await?;
    store.put("corrupt", b"not key material").await?;

    let report = envelope.rotate_master_keys().await?;
    assert_eq!(report.rewrapped, 1);
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].0, "corrupt");
    Ok(())
}

#[test]
fn envelope_rejects_invalid_data_key_length() {
    let envelope = EnvelopeEncryption::new(Arc::new(TestKms::new(&["mek"])));
    expect_err(envelope.with_data_key_length(7), "16, 24, or 32 bytes");
}

/// Rewrite the postscript of `bytes` with `edit`, as an attacker without keys could, since the
/// postscript is not authenticated.
fn edit_postscript(
    bytes: &ByteBuffer,
    edit: impl FnOnce(&mut Postscript),
) -> VortexResult<ByteBuffer> {
    let eof = bytes.len() - EOF_SIZE;
    let ps_len = usize::from(u16::from_le_bytes([bytes[eof + 2], bytes[eof + 3]]));
    let mut postscript = Postscript::read_flatbuffer_bytes(&bytes[eof - ps_len..eof])?;
    edit(&mut postscript);
    let edited = postscript.write_flatbuffer_bytes()?;
    let mut out = ByteBufferMut::copy_from(&bytes[..eof - ps_len]);
    out.extend_from_slice(&edited);
    out.extend_from_slice(&bytes[eof..eof + 2]);
    out.extend_from_slice(&u16::try_from(edited.len())?.to_le_bytes());
    out.extend_from_slice(&bytes[eof + 4..]);
    Ok(out.freeze())
}

/// Stripping the signatures of a plaintext footer would let an attacker modify the footer, so a
/// reader configured for decryption rejects it.
#[tokio::test]
async fn rejects_stripped_footer_signatures() -> VortexResult<()> {
    let bytes = write(Some(
        FileEncryption::new(footer_key())
            .with_plaintext_footer()
            .with_plaintext_fields(),
    ))
    .await?;
    let stripped = edit_postscript(&bytes, |postscript| {
        for segment in [&mut postscript.layout, &mut postscript.footer]
            .into_iter()
            .chain(postscript.dtype.as_mut())
            .chain(postscript.statistics.as_mut())
        {
            segment.encryption = EncryptionSpec::plaintext();
            segment.signature = None;
        }
    })?;
    expect_err(
        open(stripped.clone(), Some(decryption())),
        "not protected by its footer key",
    );
    // Signature verification can be skipped, but not the protection itself.
    expect_err(
        open(
            stripped.clone(),
            Some(decryption().without_footer_verification()),
        ),
        "not protected by its footer key",
    );
    // A reader without keys cannot tell, as for any plaintext footer it cannot verify.
    SESSION.open_options().open_buffer(stripped)?;
    Ok(())
}

/// Replacing the encrypted footer segments with their plaintext would let an attacker who once
/// held the footer key forge them, so a reader configured for decryption rejects it.
#[tokio::test]
async fn rejects_decrypted_footer_segments() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let eof = bytes.len() - EOF_SIZE;
    let ps_len = usize::from(u16::from_le_bytes([bytes[eof + 2], bytes[eof + 3]]));
    let postscript = Postscript::read_flatbuffer_bytes(&bytes[eof - ps_len..eof])?;
    let cipher = FileEncryption::new(footer_key())
        .cipher_for_spec(&footer_key(), &postscript.footer.encryption)?;

    let binding = postscript.binding();
    let mut decrypted = ByteBufferMut::copy_from(&bytes);
    let mut decrypt = |segment: &PostscriptSegment, role| -> VortexResult<()> {
        let start = usize::try_from(segment.offset)?;
        let end = start + segment.length as usize;
        let plaintext = cipher.decrypt(
            role,
            ByteBuffer::copy_from(&bytes[start..end]),
            segment.alignment,
        )?;
        decrypted[start..start + plaintext.len()].copy_from_slice(&plaintext);
        Ok(())
    };
    decrypt(&postscript.layout, SegmentRole::Layout)?;
    decrypt(&postscript.footer, SegmentRole::Footer(&binding))?;
    if let Some(dtype) = &postscript.dtype {
        decrypt(dtype, SegmentRole::DType)?;
    }
    if let Some(statistics) = &postscript.statistics {
        decrypt(statistics, SegmentRole::Statistics)?;
    }

    let decrypted = edit_postscript(&decrypted.freeze(), |postscript| {
        for segment in [&mut postscript.layout, &mut postscript.footer]
            .into_iter()
            .chain(postscript.dtype.as_mut())
            .chain(postscript.statistics.as_mut())
        {
            segment.encryption = EncryptionSpec::plaintext();
            segment.length -= u32::try_from(NONCE_TAG_LEN).vortex_expect("small");
        }
    })?;
    expect_err(
        open(decrypted.clone(), Some(decryption())),
        "not protected by its footer key",
    );
    // A reader without keys reads the plaintext footer.
    SESSION.open_options().open_buffer(decrypted)?;
    Ok(())
}

/// Stripping the encryption of a metadata segment would let an attacker forge its value.
#[tokio::test]
async fn rejects_stripped_metadata_encryption() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let stripped = edit_postscript(&bytes, |postscript| {
        postscript.metadata[0].segment.encryption = EncryptionSpec::plaintext();
    })?;
    expect_err(open(stripped, Some(decryption())), "is not encrypted");
    Ok(())
}

fn read_postscript(bytes: &[u8]) -> VortexResult<Postscript> {
    let eof = bytes.len() - EOF_SIZE;
    let ps_len = usize::from(u16::from_le_bytes([bytes[eof + 2], bytes[eof + 3]]));
    Postscript::read_flatbuffer_bytes(&bytes[eof - ps_len..eof])
}

/// The footer authenticates the postscript entries it does not list, so dropping them fails.
#[rstest]
#[case::metadata(|postscript: &mut Postscript| postscript.metadata.clear())]
#[case::statistics(|postscript: &mut Postscript| postscript.statistics = None)]
#[tokio::test]
async fn rejects_dropped_postscript_entries(#[case] edit: fn(&mut Postscript)) -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let edited = edit_postscript(&bytes, edit)?;
    expect_err(open(edited, Some(decryption())), "Failed to decrypt Footer");
    Ok(())
}

/// Statistics that fail to decrypt were tampered with, which fails the open.
#[tokio::test]
async fn rejects_tampered_statistics() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
    let statistics = read_postscript(&bytes)?
        .statistics
        .vortex_expect("statistics segment");
    let mut tampered = ByteBufferMut::copy_from(&bytes);
    tampered[usize::try_from(statistics.offset)?] ^= 1;
    expect_err(
        open(tampered.freeze(), Some(decryption())),
        "Failed to decrypt Statistics",
    );
    Ok(())
}

/// The standalone footer of a plaintext file can be encrypted, unless the file has plaintext
/// metadata segments, which an encrypted footer cannot reference.
#[tokio::test]
async fn encrypted_standalone_footer_of_plaintext_file() -> VortexResult<()> {
    let encryption = FileEncryption::new(footer_key());
    let with_metadata = open(write(None).await?, None)?.footer().clone();
    expect_err(
        with_metadata.into_serializer().with_encryption(&encryption),
        "plaintext metadata segments",
    );

    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array()?.to_array_stream())
        .await?;
    let bytes = buf.freeze();
    let footer = open(bytes.clone(), None)?.footer().clone();
    let footer_bytes = footer
        .into_serializer()
        .with_encryption(&encryption)?
        .serialize()?
        .iter()
        .flat_map(|buffer| buffer.as_slice().iter().copied())
        .collect::<Vec<u8>>();
    assert!(!contains(&footer_bytes, SECRET_COLUMN.as_bytes()));

    let footer = deserialize_footer(&footer_bytes, decryption()).await?;
    let file = SESSION
        .open_options()
        .with_decryption(decryption())
        .with_footer(footer)
        .open_buffer(bytes)?;
    assert_reads_all(&file).await
}

/// Swapping the fields of encrypted statistics would attribute one field's statistics to
/// another, and prune wrongly.
#[tokio::test]
async fn rejects_swapped_statistics_fields() -> VortexResult<()> {
    let encryption = FileEncryption::new(footer_key())
        .with_field_key(FieldPath::from_name(SECRET_COLUMN), "pii", column_key())
        .with_field_key(FieldPath::from_name("visible"), "pii", column_key());
    let bytes = write(Some(encryption)).await?;
    assert_eq!(
        open(bytes.clone(), Some(decryption()))?
            .footer()
            .statistics_encryption()
            .as_ref(),
        [(0, 2), (1, 2)]
    );
    let swapped = edit_postscript(&bytes, |postscript| {
        postscript.encrypted_statistics[0].fields.reverse();
    })?;
    expect_err(open(swapped, Some(decryption())), "Failed to decrypt");
    Ok(())
}

/// Invalid field keys fail the write instead of panicking.
#[rstest]
#[case::root(FieldPath::root(), FieldPath::from_name("numbers"), "root field")]
#[case::overlap(
    FieldPath::from_name(SECRET_COLUMN),
    FieldPath::from_name(SECRET_COLUMN),
    "overlap"
)]
#[tokio::test]
async fn rejects_invalid_field_keys(
    #[case] first: FieldPath,
    #[case] second: FieldPath,
    #[case] expected: &str,
) -> VortexResult<()> {
    let encryption = FileEncryption::new(footer_key())
        .with_field_key(first, "first", column_key())
        .with_field_key(second, "second", column_key());
    expect_err(write(Some(encryption)).await, expected);
    Ok(())
}

/// Readers resolve keys by their metadata, so it must not identify two different keys.
#[tokio::test]
async fn rejects_shared_key_metadata() -> VortexResult<()> {
    let other = KeyConfig::new(EncryptionKey::from([9; 32]), COLUMN_KEY_METADATA);
    let encryption =
        with_column_key(FileEncryption::new(footer_key())).with_named_key("other", other);
    expect_err(write(Some(encryption)).await, "share the same key metadata");
    Ok(())
}

/// The footer authenticates the encrypted statistics entries of the postscript.
#[tokio::test]
async fn rejects_dropped_encrypted_statistics() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    let edited = edit_postscript(&bytes, |postscript| postscript.encrypted_statistics.clear())?;
    expect_err(open(edited, Some(decryption())), "Failed to decrypt Footer");
    Ok(())
}

/// Readers can open a file without retrieving field keys, until they read the field.
#[tokio::test]
async fn field_keys_are_retrieved_on_demand() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    let counting = Arc::new(CountingKeys(Mutex::default()));
    let file = open(
        bytes,
        Some(FileDecryption::new(Arc::clone(&counting) as _).without_field_key_statistics()),
    )?;
    assert_eq!(counting.0.lock().get(COLUMN_KEY_METADATA), None);
    let stats = file.footer().statistics().vortex_expect("file statistics");
    assert!(stats.aggregates()[0].is_empty());

    assert_reads_all(&file).await?;
    assert_eq!(counting.0.lock().get(COLUMN_KEY_METADATA), Some(&1));
    Ok(())
}

/// A [`KeyRetriever`] counting the keys it retrieves.
struct CountingKeys(Mutex<HashMap<Vec<u8>, usize>>);

#[async_trait::async_trait]
impl KeyRetriever for CountingKeys {
    async fn retrieve_key(&self, key_metadata: &[u8]) -> VortexResult<EncryptionKey> {
        *self.0.lock().entry(key_metadata.to_vec()).or_default() += 1;
        keys(true).retrieve_key(key_metadata).await
    }
}

/// The footer and the segments share key resolution, so each key is retrieved once per open.
#[tokio::test]
async fn retrieves_each_key_once() -> VortexResult<()> {
    let bytes = write(Some(with_column_key(FileEncryption::new(footer_key())))).await?;
    let counting = Arc::new(CountingKeys(Mutex::default()));
    let file = SESSION
        .open_options()
        .with_decryption(FileDecryption::new(Arc::clone(&counting) as _))
        .include_metadata()
        .open_read(bytes)
        .await?;
    assert_reads_all(&file).await?;
    assert_reads_all(&file).await?;
    let counts = counting.0.lock();
    assert_eq!(counts.get(FOOTER_KEY_METADATA), Some(&1));
    assert_eq!(counts.get(COLUMN_KEY_METADATA), Some(&1));
    Ok(())
}

/// Key material claiming a cached KEK under another master key must not be served from the
/// cache, or a reader could use a master key the KMS would not let it use.
#[tokio::test]
async fn envelope_kek_cache_is_scoped_to_master_key() -> VortexResult<()> {
    use crate::flatbuffers::key_material as km;

    let kms = Arc::new(TestKms::new(&["mek-1", "mek-2"]));
    let envelope = EnvelopeEncryption::new(Arc::clone(&kms) as _);
    let key = envelope.key_config("mek-1").await?;
    // The KEK of "mek-1" is now cached; unwrapping the genuine material needs no KMS call.
    let unwraps = kms.unwraps.load(Ordering::Relaxed);
    envelope.retrieve_key(key.key_metadata()).await?;
    assert_eq!(kms.unwraps.load(Ordering::Relaxed), unwraps);

    // Copy the material, but claim that "mek-2" wraps the KEK.
    let forged = {
        let metadata = flatbuffers::root::<km::EnvelopeKeyMetadata>(key.key_metadata())?;
        let material = metadata.material().vortex_expect("inline key material");
        let mut fbb = FlatBufferBuilder::new();
        let args = km::KeyMaterialArgs {
            master_key_id: Some(fbb.create_string("mek-2")),
            wrapped_dek: Some(fbb.create_vector(material.wrapped_dek().bytes())),
            kek_id: material.kek_id().map(|id| fbb.create_vector(id.bytes())),
            wrapped_kek: material
                .wrapped_kek()
                .map(|kek| fbb.create_vector(kek.bytes())),
        };
        let material = km::KeyMaterial::create(&mut fbb, &args);
        let metadata = km::EnvelopeKeyMetadata::create(
            &mut fbb,
            &km::EnvelopeKeyMetadataArgs {
                material: Some(material),
                external_id: None,
            },
        );
        fbb.finish_minimal(metadata);
        fbb.finished_data().to_vec()
    };
    expect_err(envelope.retrieve_key(&forged).await, "failed to unwrap key");
    assert_eq!(kms.unwraps.load(Ordering::Relaxed), unwraps + 1);
    Ok(())
}
