// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_array::ArrayId;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::CompressionSessionExt;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
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
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, CompressionSession, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
/// use vortex_session::VortexSession;
///
/// let session = VortexSession::empty().with::<CompressionSession>();
///
/// // Compressor with every scheme registered on the session.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session).build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session)
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    /// The session's schemes plus those added with [`with_new_scheme`](Self::with_new_scheme).
    schemes: Vec<&'static dyn Scheme>,
    compact: bool,
    cuda_compatible: bool,
    excluded: HashSet<SchemeId>,
    allowed_encodings: Option<HashSet<ArrayId>>,
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with every scheme registered in the session's
    /// [`CompressionSession`](crate::CompressionSession).
    pub fn from_session(session: &VortexSession) -> Self {
        Self::with_schemes(session.compression().schemes().to_vec())
    }

    /// Creates a builder with no schemes registered.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    pub fn empty() -> Self {
        Self::with_schemes(Vec::new())
    }

    fn with_schemes(schemes: Vec<&'static dyn Scheme>) -> Self {
        Self {
            schemes,
            compact: false,
            cuda_compatible: false,
            excluded: HashSet::new(),
            allowed_encodings: None,
        }
    }

    /// Adds a compression scheme not registered on the session.
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to register their own schemes
    /// with the compressor. Presets, exclusions and the allowed-encodings filter still apply to
    /// it when the compressor is built.
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
    /// Pco schemes for integers and floats are included. Compact schemes already present are
    /// not added twice.
    #[cfg(feature = "zstd")]
    pub fn with_compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Excludes schemes without CUDA kernel support, keeps FSST for string compression,
    /// and adds Zstd for binary compression.
    ///
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level
    /// compression preserves binary arrays' buffer layout for zero-conversion GPU decompression,
    /// but belongs to the opt-in `zstd` edition, so callers filter the two through
    /// [`retain_allowed_encodings`](Self::retain_allowed_encodings).
    ///
    /// This preset is intended for files that will be decoded by CUDA kernels. It may choose a
    /// larger encoded representation than the default compressor. Its exclusions win over
    /// [`with_compact`](Self::with_compact), which then only contributes binary Zstd.
    pub fn only_cuda_compatible(mut self) -> Self {
        self.cuda_compatible = true;
        self
    }

    /// Removes the specified compression schemes by their [`SchemeId`].
    ///
    /// Exclusions win over schemes added by presets or [`with_new_scheme`](Self::with_new_scheme).
    pub fn exclude_schemes(mut self, ids: impl IntoIterator<Item = SchemeId>) -> Self {
        self.excluded.extend(ids);
        self
    }

    /// Retains only schemes whose produced serialized IDs all belong to `allowed`.
    ///
    /// `allowed` holds serialized IDs. The file writer passes the array IDs its enabled editions
    /// permit. Calling this more than once keeps the intersection.
    pub fn retain_allowed_encodings(mut self, allowed: &HashSet<ArrayId>) -> Self {
        self.allowed_encodings = Some(match self.allowed_encodings {
            Some(current) => current.intersection(allowed).copied().collect(),
            None => allowed.clone(),
        });
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.resolve()))
    }

    /// Resolves the configuration into the final scheme list: presets first, then exclusions,
    /// then the allowed-encodings filter.
    fn resolve(self) -> Vec<&'static dyn Scheme> {
        let mut schemes = self.schemes;
        let mut add = |scheme: &'static dyn Scheme| {
            if !schemes.iter().any(|s| s.id() == scheme.id()) {
                schemes.push(scheme);
            }
        };

        if self.compact {
            compact_schemes().into_iter().for_each(&mut add);
        }
        let mut excluded = self.excluded;
        if self.cuda_compatible {
            cuda_added_schemes().into_iter().for_each(&mut add);
            excluded.extend(cuda_excluded_schemes());
        }

        schemes.retain(|s| !excluded.contains(&s.id()));
        if let Some(allowed) = &self.allowed_encodings {
            schemes.retain(|s| s.produced_encodings().iter().all(|id| allowed.contains(id)));
        }
        schemes
    }
}

/// The schemes [`BtrBlocksCompressorBuilder::with_compact`] adds.
fn compact_schemes() -> Vec<&'static dyn Scheme> {
    vec![
        #[cfg(feature = "zstd")]
        &string::ZstdScheme,
        #[cfg(feature = "zstd")]
        &binary::ZstdScheme,
        #[cfg(feature = "pco")]
        &integer::PcoScheme,
        #[cfg(feature = "pco")]
        &float::PcoScheme,
    ]
}

/// The schemes [`BtrBlocksCompressorBuilder::only_cuda_compatible`] adds.
fn cuda_added_schemes() -> Vec<&'static dyn Scheme> {
    vec![
        #[cfg(feature = "zstd")]
        &binary::ZstdScheme,
        #[cfg(feature = "zstd")]
        &binary::ZstdBuffersScheme,
    ]
}

/// The schemes [`BtrBlocksCompressorBuilder::only_cuda_compatible`] excludes.
fn cuda_excluded_schemes() -> Vec<SchemeId> {
    // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other
    // string fragmentation and dictionary schemes still require unsupported decode paths.
    vec![
        integer::SparseScheme.id(),
        integer::IntRLEScheme.id(),
        float::ALPRDScheme.id(),
        float::FloatRLEScheme.id(),
        float::NullDominatedSparseScheme.id(),
        string::NullDominatedSparseScheme.id(),
        string::StringDictScheme.id(),
        binary::BinaryDictScheme.id(),
        // Delta now has a CUDA decode kernel, so arrays that reach the GPU already encoded with
        // it — the Delta children OnPair emits, for instance — decode there. It stays excluded
        // from this preset until GPU delta decode is benchmarked against the schemes it would
        // displace, since the preset picks encodings rather than merely decoding them.
        integer::DeltaScheme::default().id(),
        // Strings stay on FSST, so string Zstd from `with_compact` is dropped.
        #[cfg(feature = "zstd")]
        string::ZstdScheme.id(),
        #[cfg(feature = "zstd")]
        string::ZstdBuffersScheme.id(),
        #[cfg(feature = "pco")]
        integer::PcoScheme.id(),
        #[cfg(feature = "pco")]
        float::PcoScheme.id(),
    ]
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
    }

    fn ids(builder: BtrBlocksCompressorBuilder) -> Vec<SchemeId> {
        builder.resolve().iter().map(|s| s.id()).collect()
    }

    #[test]
    fn empty_starts_with_no_schemes() {
        assert!(BtrBlocksCompressorBuilder::empty().resolve().is_empty());
    }

    #[test]
    fn from_session_includes_registered_schemes() {
        let session = VortexSession::empty().with::<CompressionSession>();
        let schemes = BtrBlocksCompressorBuilder::from_session(&session).resolve();
        assert_eq!(schemes.len(), CompressionSession::default().schemes().len());

        session.register_scheme(&DELTA_SCHEME);
        let schemes = BtrBlocksCompressorBuilder::from_session(&session).resolve();
        assert_eq!(schemes.last().map(|s| s.id()), Some(DELTA_SCHEME.id()));
    }

    #[test]
    fn retain_allowed_encodings_filters_schemes() {
        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let schemes = default_builder()
            .retain_allowed_encodings(&allowed)
            .resolve();
        assert_eq!(schemes.len(), 1);
        assert_eq!(schemes[0].id(), integer::FoRScheme.id());

        let none = default_builder().retain_allowed_encodings(&HashSet::new());
        assert!(none.resolve().is_empty());
    }

    #[test]
    fn retain_allowed_encodings_intersects() {
        let for_only: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let all: HashSet<ArrayId> = CompressionSession::default()
            .schemes()
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let builder = default_builder()
            .retain_allowed_encodings(&for_only)
            .retain_allowed_encodings(&all);
        assert_eq!(ids(builder), vec![integer::FoRScheme.id()]);
    }

    #[test]
    fn retaining_all_declared_outputs_keeps_every_scheme() {
        let allowed: HashSet<ArrayId> = CompressionSession::default()
            .schemes()
            .iter()
            .flat_map(|scheme| scheme.produced_encodings())
            .collect();
        let schemes = default_builder()
            .retain_allowed_encodings(&allowed)
            .resolve();
        assert_eq!(schemes.len(), CompressionSession::default().schemes().len());
    }

    /// Filters apply to schemes added after them.
    #[test]
    fn filters_apply_to_later_schemes() {
        let excluded = default_builder()
            .exclude_schemes([DELTA_SCHEME.id()])
            .with_new_scheme(&DELTA_SCHEME);
        assert!(!ids(excluded).contains(&DELTA_SCHEME.id()));

        let allowed: HashSet<ArrayId> = [FoR.id()].into_iter().collect();
        let filtered = default_builder()
            .retain_allowed_encodings(&allowed)
            .with_new_scheme(&DELTA_SCHEME);
        assert_eq!(ids(filtered), vec![integer::FoRScheme.id()]);
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let ids = ids(default_builder().only_cuda_compatible());
        assert!(!ids.contains(&float::ALPRDScheme.id()));
    }

    /// `vortex.sparse` has no CUDA decode kernel, so no sparse scheme may survive this preset.
    #[test]
    fn cuda_compatible_excludes_every_sparse_scheme() {
        let ids = ids(default_builder().only_cuda_compatible());
        for excluded in [
            integer::SparseScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
        ] {
            assert!(!ids.contains(&excluded), "{excluded} should be excluded");
        }
    }

    #[test]
    fn cuda_compatible_uses_fsst_for_strings() {
        let ids = ids(default_builder().only_cuda_compatible());
        assert!(ids.contains(&string::FSSTScheme.id()));
        #[cfg(feature = "zstd")]
        assert!(!ids.contains(&string::ZstdScheme.id()));
    }

    #[test]
    #[cfg(feature = "pco")]
    fn cuda_compatible_excludes_pco() {
        let builder = default_builder()
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible();
        let ids = ids(builder);
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!ids.contains(&scheme));
        }
    }

    /// The presets combine the same way in either order, and CUDA's exclusions win.
    #[test]
    #[cfg(feature = "zstd")]
    fn cuda_and_compact_combine_in_either_order() {
        let cuda_first = ids(default_builder().only_cuda_compatible().with_compact());
        let compact_first = ids(default_builder().with_compact().only_cuda_compatible());
        assert_eq!(cuda_first, compact_first);
        assert_eq!(cuda_first, ids(default_builder().only_cuda_compatible()));
    }
}
