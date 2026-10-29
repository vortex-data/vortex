// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::Mutex;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use super::EncryptionSpec;
use super::FileDecryption;
use super::SegmentCipher;

/// The outcome of resolving a spec's cipher. Errors are shared, so that every segment of a spec
/// whose key is unavailable reports the same error without retrying the retriever.
type Resolution = Result<Option<Arc<SegmentCipher>>, Arc<VortexError>>;

/// Resolves the cipher of each [`EncryptionSpec`] of one file at most once, when it is first
/// needed, and shares it between the footer deserializer and the file's segment source.
pub(crate) struct CipherResolver {
    decryption: Option<FileDecryption>,
    resolutions: Mutex<HashMap<EncryptionSpec, Shared<BoxFuture<'static, Resolution>>>>,
}

impl CipherResolver {
    pub(crate) fn new(decryption: Option<FileDecryption>) -> Arc<Self> {
        Arc::new(Self {
            decryption,
            resolutions: Mutex::default(),
        })
    }

    pub(crate) fn decryption(&self) -> Option<&FileDecryption> {
        self.decryption.as_ref()
    }

    /// The pending or finished resolution of `spec`, started if this is its first request.
    fn resolution(&self, spec: &EncryptionSpec) -> Shared<BoxFuture<'static, Resolution>> {
        let mut resolutions = self.resolutions.lock();
        if let Some(resolution) = resolutions.get(spec) {
            return resolution.clone();
        }
        let decryption = self.decryption.clone();
        let owned = spec.clone();
        let resolution = async move {
            SegmentCipher::resolve(&owned, decryption.as_ref())
                .await
                .map(|cipher| cipher.map(Arc::new))
                .map_err(Arc::new)
        }
        .boxed()
        .shared();
        resolutions.insert(spec.clone(), resolution.clone());
        resolution
    }

    /// The cipher of `spec`, or `None` if it describes plaintext segments.
    pub(crate) async fn cipher(
        &self,
        spec: &EncryptionSpec,
    ) -> VortexResult<Option<Arc<SegmentCipher>>> {
        self.resolution(spec).await.map_err(VortexError::from)
    }

    /// The cipher of `spec` if its resolution finished, or `None` if it has not.
    pub(crate) fn resolved(
        &self,
        spec: &EncryptionSpec,
    ) -> Option<VortexResult<Option<Arc<SegmentCipher>>>> {
        self.resolutions
            .lock()
            .get(spec)
            .and_then(Shared::peek)
            .map(|resolution| resolution.clone().map_err(VortexError::from))
    }
}
