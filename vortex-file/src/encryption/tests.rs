// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::sync::Arc;
use std::sync::LazyLock;

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
use vortex_array::flatbuffers::ReadFlatBuffer;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSession;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::*;
use crate::DeserializeStep;
use crate::EOF_SIZE;
use crate::Footer;
use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
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
const SECRET: &str = "a-secret-string-value";
const SECRET_COLUMN: &str = "secret_column_name";
const VISIBLE: &str = "a-visible-string-value";
const AAD: &[u8] = b"s3://bucket/table/part-0.vortex";

fn footer_key() -> KeyConfig {
    KeyConfig::new(EncryptionKey::from([42; 32]), FOOTER_KEY_METADATA)
}

/// Resolves the footer key.
fn keys() -> Arc<dyn KeyRetriever> {
    Arc::new(move |key_metadata: &[u8]| match key_metadata {
        FOOTER_KEY_METADATA => Ok(footer_key().key().clone()),
        _ => vortex_bail!("access denied to key {key_metadata:?}"),
    })
}

fn decryption() -> FileDecryption {
    FileDecryption::new(keys())
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
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
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
    assert_reads_all(&file).await
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

#[tokio::test]
async fn aad_prefix() -> VortexResult<()> {
    let encryption = FileEncryption::new(footer_key()).with_aad_prefix(AAD);
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

/// A [`KeyRetriever`] counting the keys it retrieves.
struct CountingKeys(Mutex<HashMap<Vec<u8>, usize>>);

#[async_trait::async_trait]
impl KeyRetriever for CountingKeys {
    async fn retrieve_key(&self, key_metadata: &[u8]) -> VortexResult<EncryptionKey> {
        *self.0.lock().entry(key_metadata.to_vec()).or_default() += 1;
        keys().retrieve_key(key_metadata).await
    }
}

/// The footer and the segments share key resolution, so each key is retrieved once per open.
#[tokio::test]
async fn retrieves_each_key_once() -> VortexResult<()> {
    let bytes = write(Some(FileEncryption::new(footer_key()))).await?;
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
    Ok(())
}
