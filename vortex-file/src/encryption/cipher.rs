// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encryption, decryption and signing of individual segments.

use std::sync::Arc;

use aes_gcm::AeadInOut;
use aes_gcm::Aes128Gcm;
use aes_gcm::Aes256Gcm;
use aes_gcm::AesGcm;
use aes_gcm::KeyInit;
use aes_gcm::Nonce;
use aes_gcm::Tag;
use aes_gcm::aead::consts::U12;
use aes_gcm::aes::Aes192;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::EncryptionKey;
use super::EncryptionSpec;
use super::FileDecryption;
use super::fill_random;

pub(crate) const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// Length in bytes of the `nonce || tag` trailer of an encrypted segment.
pub(crate) const NONCE_TAG_LEN: usize = NONCE_LEN + TAG_LEN;

type Aes192Gcm = AesGcm<Aes192, U12>;

/// Identifies what an encrypted segment holds, and is authenticated alongside its ciphertext.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SegmentRole<'a> {
    /// A segment of the segment map, by index.
    Data(u32),
    DType,
    Layout,
    Statistics,
    /// The footer, bound to the postscript entries it cannot authenticate otherwise, as encoded by
    /// [`postscript_binding`](crate::footer::postscript::postscript_binding).
    Footer(&'a [u8]),
    /// A user-defined metadata segment, by key.
    Metadata(&'a str),
}

impl SegmentRole<'_> {
    fn write_aad(&self, aad: &mut Vec<u8>) {
        match self {
            SegmentRole::Data(index) => {
                aad.push(0);
                aad.extend_from_slice(&index.to_le_bytes());
            }
            SegmentRole::DType => aad.push(1),
            SegmentRole::Layout => aad.push(2),
            SegmentRole::Statistics => aad.push(3),
            SegmentRole::Footer(binding) => {
                aad.push(4);
                aad.extend_from_slice(binding);
            }
            SegmentRole::Metadata(key) => {
                aad.push(5);
                aad.extend_from_slice(key.as_bytes());
            }
        }
    }
}

/// AES-GCM with any of the supported key lengths.
#[derive(Clone)]
pub(crate) enum Aead {
    Aes128(Aes128Gcm),
    Aes192(Aes192Gcm),
    Aes256(Aes256Gcm),
}

macro_rules! dispatch {
    ($aead:expr, $cipher:ident => $body:expr) => {
        match $aead {
            Aead::Aes128($cipher) => $body,
            Aead::Aes192($cipher) => $body,
            Aead::Aes256($cipher) => $body,
        }
    };
}

impl Aead {
    pub(crate) fn new(key: &EncryptionKey) -> VortexResult<Self> {
        let bytes = key.as_bytes();
        let invalid = |_| vortex_err!("Invalid AES key length {}", bytes.len());
        Ok(match bytes.len() {
            16 => Self::Aes128(Aes128Gcm::new_from_slice(bytes).map_err(invalid)?),
            24 => Self::Aes192(Aes192Gcm::new_from_slice(bytes).map_err(invalid)?),
            32 => Self::Aes256(Aes256Gcm::new_from_slice(bytes).map_err(invalid)?),
            len => vortex_bail!("Invalid AES key length {len}"),
        })
    }

    /// Encrypt `buffer` in place, returning `nonce || tag`.
    pub(crate) fn seal(&self, aad: &[u8], buffer: &mut [u8]) -> VortexResult<[u8; NONCE_TAG_LEN]> {
        let mut nonce = [0u8; NONCE_LEN];
        fill_random(&mut nonce)?;
        let tag = dispatch!(self, cipher => cipher.encrypt_inout_detached(&Nonce::from(nonce), aad, buffer.into()))
            .map_err(|_| vortex_err!("AES-GCM encryption failed"))?;
        let mut out = [0u8; NONCE_TAG_LEN];
        out[..NONCE_LEN].copy_from_slice(&nonce);
        out[NONCE_LEN..].copy_from_slice(tag.as_slice());
        Ok(out)
    }

    /// Decrypt `buffer` in place, authenticating it against `nonce || tag`.
    pub(crate) fn open(&self, aad: &[u8], buffer: &mut [u8], nonce_tag: &[u8]) -> Result<(), ()> {
        if nonce_tag.len() != NONCE_TAG_LEN {
            return Err(());
        }
        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&nonce_tag[..NONCE_LEN]);
        let mut tag = [0u8; TAG_LEN];
        tag.copy_from_slice(&nonce_tag[NONCE_LEN..]);
        let (nonce, tag) = (Nonce::from(nonce), Tag::from(tag));
        dispatch!(self, cipher => cipher.decrypt_inout_detached(&nonce, aad, buffer.into(), &tag))
            .map_err(|_| ())
    }
}

/// Encrypts, decrypts and signs the segments of one [`EncryptionSpec`].
#[derive(Clone)]
pub(crate) struct SegmentCipher {
    spec: EncryptionSpec,
    aead: Aead,
    /// The additional authenticated data shared by every segment, without the segment's role.
    aad: Arc<[u8]>,
}

impl SegmentCipher {
    pub(crate) fn new(
        spec: EncryptionSpec,
        key: EncryptionKey,
        aad_prefix: &[u8],
    ) -> VortexResult<Self> {
        // Length-prefix the application-defined prefix, so that it cannot run into the bytes that
        // follow it.
        let prefix_len = u32::try_from(aad_prefix.len()).map_err(|_| {
            vortex_err!(
                "AAD prefix of {} bytes is longer than the maximum of {} bytes",
                aad_prefix.len(),
                u32::MAX
            )
        })?;
        let mut aad = Vec::with_capacity(4 + aad_prefix.len() + spec.aad_file_unique.len());
        aad.extend_from_slice(&prefix_len.to_le_bytes());
        aad.extend_from_slice(aad_prefix);
        aad.extend_from_slice(&spec.aad_file_unique);
        Ok(Self {
            spec,
            aead: Aead::new(&key)?,
            aad: aad.into(),
        })
    }

    /// Resolve the cipher for `spec`, or `None` if it describes plaintext segments.
    pub(crate) async fn resolve(
        spec: &EncryptionSpec,
        decryption: Option<&FileDecryption>,
    ) -> VortexResult<Option<Self>> {
        if !spec.is_encrypted() {
            return Ok(None);
        }
        let decryption = decryption.ok_or_else(|| {
            vortex_err!(
                "Vortex file is encrypted, but no FileDecryption was provided to \
                 VortexOpenOptions::with_decryption"
            )
        })?;
        let aad_prefix = match (decryption.aad_prefix.as_deref(), spec.requires_aad_prefix) {
            (Some(supplied), true) => supplied,
            (None, false) => &[],
            (Some(_), false) => vortex_bail!(
                "An AAD prefix was supplied with FileDecryption::with_aad_prefix, but the Vortex \
                 file is not bound to one"
            ),
            (None, true) => vortex_bail!(
                "Vortex file requires an AAD prefix, supplied with FileDecryption::with_aad_prefix"
            ),
        };
        let key = decryption.keys.retrieve_key(&spec.key_metadata).await?;
        Ok(Some(Self::new(spec.clone(), key, aad_prefix)?))
    }

    pub(crate) fn spec(&self) -> &EncryptionSpec {
        &self.spec
    }

    /// The number of bytes an encrypted segment occupies beyond its plaintext.
    pub(crate) fn overhead(&self) -> usize {
        NONCE_TAG_LEN
    }

    fn aad(&self, role: SegmentRole<'_>) -> Vec<u8> {
        let mut aad = Vec::with_capacity(self.aad.len() + 9);
        aad.extend_from_slice(&self.aad);
        role.write_aad(&mut aad);
        aad
    }

    /// Encrypt the concatenation of `buffers` into a single buffer aligned to `alignment`.
    ///
    /// The result is `ciphertext || nonce || tag`, with a ciphertext of the plaintext's length.
    pub(crate) fn encrypt(
        &self,
        role: SegmentRole<'_>,
        buffers: &[ByteBuffer],
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let len = buffers.iter().map(ByteBuffer::len).sum::<usize>();
        let mut out = ByteBufferMut::with_capacity_aligned(len + self.overhead(), alignment);
        for buffer in buffers {
            out.extend_from_slice(buffer.as_slice());
        }
        let nonce_tag = self
            .aead
            .seal(&self.aad(role), out.as_mut_slice())
            .map_err(|e| vortex_err!("Failed to encrypt {role:?} segment: {e}"))?;
        out.extend_from_slice(&nonce_tag);
        Ok(out.freeze())
    }

    /// Decrypt and authenticate an encrypted segment, returning a plaintext aligned to
    /// `alignment`.
    ///
    /// The ciphertext is decrypted in place when `encrypted` is not shared.
    pub(crate) fn decrypt(
        &self,
        role: SegmentRole<'_>,
        encrypted: ByteBuffer,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let overhead = self.overhead();
        let Some(len) = encrypted.len().checked_sub(overhead) else {
            vortex_bail!(
                "Encrypted {role:?} segment of {} bytes is shorter than the {overhead} bytes of \
                 encryption overhead",
                encrypted.len()
            );
        };
        let mut trailer = [0u8; NONCE_TAG_LEN];
        trailer.copy_from_slice(&encrypted[len..]);

        let mut plaintext = match encrypted.try_into_mut() {
            Ok(buffer) if buffer.alignment().is_aligned_to(alignment) => buffer,
            Ok(buffer) => ByteBufferMut::copy_from_aligned(buffer, alignment),
            Err(buffer) => ByteBufferMut::copy_from_aligned(buffer, alignment),
        };
        plaintext.truncate(len);
        self.aead
            .open(&self.aad(role), plaintext.as_mut_slice(), &trailer)
            .map_err(|()| {
                vortex_err!(
                    "Failed to decrypt {role:?} segment: the key or AAD prefix is wrong, or the \
                     segment was modified"
                )
            })?;
        Ok(plaintext.freeze())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::encryption::EncryptionAlgorithm;
    use crate::encryption::KeyConfig;

    fn cipher(key_len: usize) -> VortexResult<SegmentCipher> {
        KeyConfig::new(
            EncryptionKey::try_from_slice(&vec![7; key_len])?,
            b"key-1".as_slice(),
        )
        .file_cipher(EncryptionAlgorithm::AesGcm, None)
    }

    #[rstest]
    #[case::aes_128(16)]
    #[case::aes_192(24)]
    #[case::aes_256(32)]
    fn roundtrip(#[case] key_len: usize) -> VortexResult<()> {
        let cipher = cipher(key_len)?;
        let buffers = [
            ByteBuffer::copy_from(b"hello "),
            ByteBuffer::copy_from(b"world"),
        ];
        let alignment = Alignment::new(64);

        for role in [SegmentRole::Data(3), SegmentRole::Footer(b"")] {
            let encrypted = cipher.encrypt(role, &buffers, alignment)?;
            assert_eq!(encrypted.len(), 11 + cipher.overhead());
            assert_ne!(&encrypted[..11], b"hello world");

            let decrypted = cipher.decrypt(role, encrypted, alignment)?;
            assert_eq!(decrypted.as_slice(), b"hello world");
            assert!(decrypted.alignment().is_aligned_to(alignment));
        }
        Ok(())
    }

    #[test]
    fn rejects_other_role() -> VortexResult<()> {
        let cipher = cipher(32)?;
        let encrypted = cipher.encrypt(
            SegmentRole::Data(3),
            &[ByteBuffer::copy_from(b"data")],
            Alignment::none(),
        )?;
        let err = cipher
            .decrypt(SegmentRole::Data(4), encrypted, Alignment::none())
            .unwrap_err();
        assert!(err.to_string().contains("Failed to decrypt"), "{err}");
        Ok(())
    }

    #[test]
    fn rejects_other_file() -> VortexResult<()> {
        // Same key and key metadata, but a different file, and so different per-file AAD bytes.
        let encrypted = cipher(32)?.encrypt(
            SegmentRole::Footer(b""),
            &[ByteBuffer::copy_from(b"footer")],
            Alignment::none(),
        )?;
        assert!(
            cipher(32)?
                .decrypt(SegmentRole::Footer(b""), encrypted, Alignment::none())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_wrong_key() -> VortexResult<()> {
        let cipher = cipher(32)?;
        let encrypted = cipher.encrypt(
            SegmentRole::Layout,
            &[ByteBuffer::copy_from(b"layout")],
            Alignment::none(),
        )?;
        let wrong = SegmentCipher::new(cipher.spec().clone(), EncryptionKey::from([8; 32]), b"")?;
        assert!(
            wrong
                .decrypt(SegmentRole::Layout, encrypted, Alignment::none())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_truncated_segment() -> VortexResult<()> {
        let err = cipher(32)?
            .decrypt(
                SegmentRole::DType,
                ByteBuffer::copy_from([0u8; NONCE_TAG_LEN - 1]),
                Alignment::none(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("shorter"), "{err}");
        Ok(())
    }
}
