// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::CompressionSessionExt;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::permission_set::PermissionSet;
use crate::schemes::binary;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;

/// Delta, kept out of the default [`CompressionSession`](crate::CompressionSession) schemes
/// because it is slower to decompress than the schemes that would otherwise win. Callers that
/// want it opt in with [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme).
///
/// TODO(robert): Register it by default once we have scheme filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// [`from_session`](Self::from_session) starts from the schemes registered in the session's
/// [`CompressionSession`](crate::CompressionSession), in registration order. Feature-gated
/// schemes (Pco, Zstd) are not registered by default and must be registered on the session, or
/// added explicitly via [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) or
/// `with_compact` when the `zstd` feature is enabled.
///
/// The builder also tracks which serialized array IDs its schemes may produce, taken from the
/// session's registered arrays and enabled editions. [`build`](Self::build) drops every scheme
/// that produces a disallowed ID.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, CompressionSession, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
/// use vortex_session::VortexSession;
///
/// let session = VortexSession::empty().with::<CompressionSession>();
///
/// // Compressor with the session's schemes, restricted to the encodings it allows. This session
/// // enables no editions, so every scheme is dropped.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session).build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session)
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: Vec<&'static dyn Scheme>,
    allowed: PermissionSet,
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with every scheme registered in the session's
    /// [`CompressionSession`](crate::CompressionSession), allowing the serialized IDs registered
    /// in the session and permitted by its enabled editions.
    pub fn from_session(session: &VortexSession) -> Self {
        Self {
            schemes: session.compression().schemes().to_vec(),
            allowed: PermissionSet::from_session(session),
        }
    }

    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    /// Every serialized ID is allowed.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            allowed: PermissionSet::all(),
        }
    }

    /// Adds a compression scheme not registered on the session.
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to register their own schemes
    /// with the compressor.
    ///
    /// # Panics
    ///
    /// Panics if a scheme with the same [`SchemeId`] is already present.
    pub fn with_new_scheme(mut self, scheme: &'static dyn Scheme) -> Self {
        assert!(
            !self.schemes.iter().any(|s| s.id() == scheme.id()),
            "scheme {:?} is already present in the builder",
            scheme.id(),
        );

        self.schemes.push(scheme);
        self
    }

    /// Adds compact encoding schemes (Zstd for strings and binary, Pco for numerics).
    ///
    /// This provides better compression ratios than the default, especially for floating-point
    /// heavy datasets. Requires the `zstd` feature. When the `pco` feature is also enabled,
    /// Pco schemes for integers and floats are included.
    ///
    /// # Panics
    ///
    /// Panics if any of the compact schemes are already present.
    #[cfg(feature = "zstd")]
    pub fn with_compact(self) -> Self {
        let builder = self
            .with_new_scheme(&string::ZstdScheme)
            .with_new_scheme(&binary::ZstdScheme);

        #[cfg(feature = "pco")]
        let builder = builder
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme);

        builder
    }

    /// Excludes schemes without CUDA kernel support, keeps FSST for string compression,
    /// and adds Zstd for binary compression.
    ///
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level
    /// compression preserves binary arrays' buffer layout for zero-conversion GPU decompression,
    /// but belongs to the opt-in `zstd` edition, so the session's enabled editions decide which
    /// of the two [`build`](Self::build) keeps.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(self) -> Self {
        // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
        // string fragmentation and dictionary schemes still require unsupported decode paths.
        #[cfg_attr(not(any(feature = "pco", feature = "zstd")), allow(unused_mut))]
        let mut excluded: Vec<SchemeId> = vec![
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::ALPRDScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
            string::StringDictScheme.id(),
            binary::BinaryDictScheme.id(),
        ];
        // Delta now has a CUDA decode kernel, so arrays that reach the GPU already encoded with
        // it — the Delta children OnPair emits, for instance — decode there. It stays excluded
        // from this preset until GPU delta decode is benchmarked against the schemes it would
        // displace, since the preset picks encodings rather than merely decoding them.
        excluded.push(integer::DeltaScheme::default().id());
        #[cfg(feature = "pco")]
        excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
        let builder = self.exclude_schemes(excluded);

        #[cfg(feature = "zstd")]
        let builder = builder
            .with_new_scheme(&binary::ZstdScheme)
            .with_new_scheme(&binary::ZstdBuffersScheme);

        builder
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        let ids: HashSet<_> = ids.into_iter().collect();
        self.schemes.retain(|s| !ids.contains(&s.id()));
        self
    }

    /// Allows serialized IDs the session's enabled editions do not permit, still requiring them
    /// to be registered in the session.
    pub fn disable_editions(mut self) -> Self {
        self.allowed.blacklist.clear();
        self
    }

    /// Allows every serialized ID, ignoring the session's registered arrays and enabled editions.
    pub fn unrestricted(mut self) -> Self {
        self.allowed = PermissionSet::all();
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`] from the schemes whose produced serialized
    /// IDs are all allowed.
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.allowed_schemes()))
    }

    fn allowed_schemes(&self) -> Vec<&'static dyn Scheme> {
        self.schemes
            .iter()
            .copied()
            .filter(|s| {
                s.produced_encodings()
                    .iter()
                    .all(|id| self.allowed.is_allowed(id))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VTable;
    use vortex_fastlanes::FoR;

    use super::*;
    use crate::CompressionSession;

    fn default_builder() -> BtrBlocksCompressorBuilder {
        BtrBlocksCompressorBuilder::from_session(
            &VortexSession::empty().with::<CompressionSession>(),
        )
        .unrestricted()
    }

    #[test]
    fn empty_starts_with_no_schemes() {
        let builder = BtrBlocksCompressorBuilder::empty();
        assert!(builder.schemes.is_empty());
    }

    #[test]
    fn from_session_includes_registered_schemes() {
        let session = VortexSession::empty().with::<CompressionSession>();
        let builder = BtrBlocksCompressorBuilder::from_session(&session);
        assert_eq!(
            builder.schemes.len(),
            CompressionSession::default().schemes().len()
        );

        session.register_scheme(&DELTA_SCHEME);
        let builder = BtrBlocksCompressorBuilder::from_session(&session);
        assert_eq!(
            builder.schemes.last().map(|s| s.id()),
            Some(DELTA_SCHEME.id())
        );
    }

    #[test]
    fn from_session_without_enabled_editions_allows_nothing() {
        let session = vortex_array::array_session();
        vortex_fastlanes::initialize(&session);
        let builder = BtrBlocksCompressorBuilder::from_session(&session);
        assert!(builder.allowed_schemes().is_empty());
    }

    #[test]
    fn from_session_excludes_unregistered_encodings() {
        let session = vortex_array::array_session();
        let builder = BtrBlocksCompressorBuilder::from_session(&session).disable_editions();
        assert!(
            !builder
                .allowed_schemes()
                .iter()
                .any(|s| s.id() == integer::FoRScheme.id())
        );
    }

    #[test]
    fn disable_editions_allows_registered_encodings() {
        let session = vortex_array::array_session();
        vortex_fastlanes::initialize(&session);
        let builder = BtrBlocksCompressorBuilder::from_session(&session).disable_editions();
        assert!(
            builder
                .allowed_schemes()
                .iter()
                .any(|s| s.id() == integer::FoRScheme.id())
        );
    }

    #[test]
    fn unrestricted_allows_every_scheme() {
        let session = vortex_array::array_session();
        let builder = BtrBlocksCompressorBuilder::from_session(&session).unrestricted();
        assert_eq!(
            builder.allowed_schemes().len(),
            CompressionSession::default().schemes().len()
        );
    }

    #[test]
    fn blacklist_overrides_allowlist() {
        let mut builder = default_builder();
        builder.allowed.blacklist.insert(FoR.id());
        let schemes = builder.allowed_schemes();
        assert!(!schemes.iter().any(|s| s.id() == integer::FoRScheme.id()));
    }

    #[test]
    fn empty_allows_all() {
        let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&integer::FoRScheme);
        assert_eq!(builder.allowed_schemes().len(), 1);
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let builder = default_builder().only_cuda_compatible();
        assert!(
            !builder
                .schemes
                .iter()
                .any(|s| s.id() == float::ALPRDScheme.id())
        );
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let builder = default_builder().only_cuda_compatible();
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(
                !builder.schemes.iter().any(|s| s.id() == excluded),
                "{excluded} should be excluded"
            );
        }
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let builder = default_builder().only_cuda_compatible();
        assert!(
            builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::FSSTScheme.id())
        );
        #[cfg(feature = "zstd")]
        assert!(
            !builder
                .schemes
                .iter()
                .any(|scheme| scheme.id() == string::ZstdScheme.id())
        );
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let builder = default_builder()
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!builder.schemes.iter().any(|s| s.id() == scheme));
        }
    }
}
