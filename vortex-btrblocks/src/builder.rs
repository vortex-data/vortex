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
use crate::allowed_ids::AllowedIds;
use crate::schemes::binary;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;

/// The preset a [`BtrBlocksCompressorBuilder`] builds with.
///
/// Every mode except [`All`](Self::All) excludes some schemes in
/// [`build`](BtrBlocksCompressorBuilder::build).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompressionMode {
    /// Excludes nothing. Set by [`empty`](BtrBlocksCompressorBuilder::empty).
    All,
    /// Excludes Zstd and Pco. Set by [`from_session`](BtrBlocksCompressorBuilder::from_session).
    Default,
    /// Excludes buffer-level Zstd, keeping Zstd for strings and binary and Pco for numerics.
    /// Set by [`with_compact`](BtrBlocksCompressorBuilder::with_compact).
    Compact,
    /// Excludes schemes without CUDA kernel support, keeping FSST for strings and both Zstd
    /// schemes for binary. Set by
    /// [`only_cuda_compatible`](BtrBlocksCompressorBuilder::only_cuda_compatible).
    Cuda,
}

impl CompressionMode {
    /// Returns the schemes [`build`](BtrBlocksCompressorBuilder::build) drops in this mode.
    fn excluded_schemes(self) -> Vec<SchemeId> {
        let mut excluded = Vec::new();
        match self {
            Self::All => {}
            Self::Default => {
                #[cfg(feature = "zstd")]
                excluded.extend([
                    string::ZstdScheme.id(),
                    binary::ZstdScheme.id(),
                    binary::ZstdBuffersScheme.id(),
                ]);
                #[cfg(feature = "pco")]
                excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
            }
            Self::Compact => {
                #[cfg(feature = "zstd")]
                excluded.push(binary::ZstdBuffersScheme.id());
            }
            Self::Cuda => {
                // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
                // string fragmentation and dictionary schemes still require unsupported decode
                // paths. Delta has a CUDA decode kernel, but stays excluded until GPU delta
                // decode is benchmarked against the schemes it would displace.
                excluded.extend([
                    integer::DeltaScheme::default().id(),
                    integer::SparseScheme.id(),
                    integer::IntRLEScheme.id(),
                    float::ALPRDScheme.id(),
                    float::FloatRLEScheme.id(),
                    float::NullDominatedSparseScheme.id(),
                    string::NullDominatedSparseScheme.id(),
                    string::StringDictScheme.id(),
                    binary::BinaryDictScheme.id(),
                ]);
                // Both binary Zstd schemes are kept. Buffer-level compression preserves binary
                // arrays' buffer layout for zero-conversion GPU decompression, but belongs to the
                // opt-in `zstd` edition, so the session's enabled editions decide which of the
                // two is kept.
                #[cfg(feature = "zstd")]
                excluded.push(string::ZstdScheme.id());
                #[cfg(feature = "pco")]
                excluded.extend([integer::PcoScheme.id(), float::PcoScheme.id()]);
            }
        }
        excluded
    }
}

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// [`from_session`](Self::from_session) starts from the schemes registered in the session's
/// [`CompressionSession`](crate::CompressionSession), in registration order, and
/// [`build`](Self::build) excludes some of them: by default Zstd and Pco.
/// [`with_compact`](Self::with_compact) and [`only_cuda_compatible`](Self::only_cuda_compatible)
/// switch to a different preset.
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
    allowed: AllowedIds,
    mode: CompressionMode,
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with every scheme registered in the session's
    /// [`CompressionSession`](crate::CompressionSession), allowing the serialized IDs registered
    /// in the session and permitted by its enabled editions.
    pub fn from_session(session: &VortexSession) -> Self {
        Self {
            schemes: session.compression().schemes().to_vec(),
            allowed: AllowedIds::from_session(session),
            mode: CompressionMode::Default,
        }
    }

    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    /// Every added scheme and every serialized ID is allowed.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            allowed: AllowedIds::all(),
            mode: CompressionMode::All,
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

    /// Switches to the compact preset, keeping Zstd for strings and binary and Pco for numerics.
    ///
    /// This provides better compression ratios than the default, especially for floating-point
    /// heavy datasets. The Zstd and Pco schemes are only registered with the `zstd` and `pco`
    /// features.
    pub fn with_compact(mut self) -> Self {
        self.mode = CompressionMode::Compact;
        self
    }

    /// Switches to the CUDA preset, excluding schemes without CUDA kernel support, keeping FSST
    /// for string compression and Zstd for binary compression.
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor.
    pub fn only_cuda_compatible(mut self) -> Self {
        self.mode = CompressionMode::Cuda;
        self
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
        self.allowed.editions = None;
        self
    }

    /// Allows serialized IDs regardless of the session's registered arrays and enabled editions.
    ///
    /// Serialized IDs excluded by a preset stay excluded.
    pub fn unrestricted(mut self) -> Self {
        self.allowed.registered = None;
        self.allowed.editions = None;
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`] from the schemes that its preset does not
    /// exclude and whose produced serialized IDs are all allowed.
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.allowed_schemes()))
    }

    fn allowed_schemes(&self) -> Vec<&'static dyn Scheme> {
        let excluded: HashSet<SchemeId> = self.mode.excluded_schemes().into_iter().collect();
        self.schemes
            .iter()
            .copied()
            .filter(|s| !excluded.contains(&s.id()))
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
    }

    #[test]
    fn delta_is_excluded_only_by_cuda() {
        let has_delta = |builder: &BtrBlocksCompressorBuilder| {
            builder
                .allowed_schemes()
                .iter()
                .any(|s| s.id() == integer::DeltaScheme::default().id())
        };
        assert!(has_delta(&default_builder()));
        assert!(has_delta(&default_builder().with_compact()));
        assert!(!has_delta(&default_builder().only_cuda_compatible()));
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
        let excluded = CompressionMode::Default.excluded_schemes();
        assert_eq!(
            builder.allowed_schemes().len(),
            CompressionSession::default()
                .schemes()
                .iter()
                .filter(|s| !excluded.contains(&s.id()))
                .count()
        );
    }

    #[test]
    fn excluded_encodings_survive_lifting_restrictions() {
        let session = vortex_array::array_session();
        vortex_fastlanes::initialize(&session);
        for mut builder in [
            BtrBlocksCompressorBuilder::from_session(&session),
            BtrBlocksCompressorBuilder::empty().with_new_scheme(&integer::FoRScheme),
        ] {
            builder.allowed.excluded.insert(FoR.id());
            for builder in [builder.clone().disable_editions(), builder.unrestricted()] {
                assert!(
                    !builder
                        .allowed_schemes()
                        .iter()
                        .any(|s| s.id() == integer::FoRScheme.id())
                );
            }
        }
    }

    #[test]
    fn empty_allows_all() {
        let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&integer::FoRScheme);
        assert_eq!(builder.allowed_schemes().len(), 1);
    }

    /// `empty()` keeps schemes that the default preset would exclude.
    #[test]
    #[cfg(feature = "zstd")]
    fn empty_keeps_schemes_the_default_preset_excludes() {
        let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&string::ZstdScheme);
        assert_eq!(builder.allowed_schemes().len(), 1);
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let schemes = default_builder().only_cuda_compatible().allowed_schemes();
        assert!(!schemes.iter().any(|s| s.id() == float::ALPRDScheme.id()));
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let schemes = default_builder().only_cuda_compatible().allowed_schemes();
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(
                !schemes.iter().any(|s| s.id() == excluded),
                "{excluded} should be excluded"
            );
        }
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let schemes = default_builder().only_cuda_compatible().allowed_schemes();
        assert!(
            schemes
                .iter()
                .any(|scheme| scheme.id() == string::FSSTScheme.id())
        );
        #[cfg(feature = "zstd")]
        assert!(
            !schemes
                .iter()
                .any(|scheme| scheme.id() == string::ZstdScheme.id())
        );
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let schemes = default_builder().only_cuda_compatible().allowed_schemes();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!schemes.iter().any(|s| s.id() == scheme));
        }
    }
}
