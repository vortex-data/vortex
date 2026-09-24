// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use vortex_array::ArrayId;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::AllowedEncodings;
use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::schemes::binary;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::session::CompressionSessionExt;

/// Delta, kept out of [`DEFAULT_SCHEMES`](crate::DEFAULT_SCHEMES) because it is slower to
/// decompress than the schemes that would otherwise win. Callers that want it opt in with
/// [`with_new_scheme`](BtrBlocksCompressorBuilder::with_new_scheme) and permit `fastlanes.delta`.
///
/// TODO(robert): Return it to [`DEFAULT_SCHEMES`](crate::DEFAULT_SCHEMES) once we have scheme
/// filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// Builder for creating configured [`BtrBlocksCompressor`] instances.
///
/// A builder starts from the schemes registered on a session, see
/// [`CompressionSession`](crate::CompressionSession), and permits the serialized IDs that session's
/// enabled editions allow. Feature-gated schemes (Pco, Zstd) are
/// not registered by default and must be added explicitly via
/// [`with_new_scheme`](Self::with_new_scheme) or `with_compact` when the `zstd` feature is enabled.
///
/// Permissions apply in [`build`](Self::build): a scheme is kept only if every serialized ID it
/// declares is permitted, including schemes added after the permissions were changed.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressorBuilder, Scheme, SchemeExt};
/// use vortex_btrblocks::schemes::integer::IntDictScheme;
///
/// let session = vortex_array::array_session();
///
/// // Every registered scheme; without enabled editions, permit everything they declare.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session)
///     .allow_all_encodings()
///     .build();
///
/// // Remove specific schemes.
/// let compressor = BtrBlocksCompressorBuilder::from_session(&session)
///     .allow_all_encodings()
///     .exclude_schemes([IntDictScheme.id()])
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BtrBlocksCompressorBuilder {
    schemes: Vec<&'static dyn Scheme>,
    /// The serialized IDs the built compressor may write.
    allowed_encodings: AllowedEncodings,
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with the schemes registered on `session`, permitting the array
    /// encodings its enabled editions allow.
    pub fn from_session(session: &VortexSession) -> Self {
        Self {
            schemes: session.registered_schemes(),
            allowed_encodings: AllowedEncodings::only(
                session.enabled_component_ids(ComponentKind::Array),
            ),
        }
    }

    /// Creates a builder with no schemes registered and no serialized IDs permitted.
    ///
    /// Useful when the caller wants explicit, scheme-by-scheme control over the compressor.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
            allowed_encodings: AllowedEncodings::default(),
        }
    }

    /// Adds a compression scheme that is not registered on the session.
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to register their own schemes
    /// with the compressor. Like every scheme, it is dropped at [`build`](Self::build) unless all
    /// of its declared serialized IDs are permitted.
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
    /// but belongs to the opt-in `zstd` edition, so the permitted serialized IDs decide which of
    /// the two survives [`build`](Self::build).
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

    /// Permits additional serialized IDs.
    pub fn allow_encodings(mut self, ids: impl IntoIterator<Item = ArrayId>) -> Self {
        self.allowed_encodings.allow(ids);
        self
    }

    /// Permits every serialized ID that is not explicitly disallowed.
    ///
    /// Use for in-memory compression, where no editions restrict what a file may contain.
    pub fn allow_all_encodings(mut self) -> Self {
        self.allowed_encodings.allow_all();
        self
    }

    /// Withdraws permission for serialized IDs.
    ///
    /// Schemes declaring any of them are dropped at [`build`](Self::build), even after
    /// [`allow_all_encodings`](Self::allow_all_encodings).
    pub fn disallow_encodings(mut self, ids: impl IntoIterator<Item = ArrayId>) -> Self {
        self.allowed_encodings.disallow(ids);
        self
    }

    /// The serialized IDs the built compressor may write.
    pub fn allowed_encodings(&self) -> &AllowedEncodings {
        &self.allowed_encodings
    }

    /// Builds the configured [`BtrBlocksCompressor`].
    ///
    /// Keeps only the schemes whose declared serialized IDs are all permitted.
    pub fn build(self) -> BtrBlocksCompressor {
        BtrBlocksCompressor(CascadingCompressor::new(self.permitted_schemes()))
    }

    fn permitted_schemes(self) -> Vec<&'static dyn Scheme> {
        let allowed = &self.allowed_encodings;
        self.schemes
            .iter()
            .copied()
            .filter(|scheme| allowed.permits_all(&scheme.produced_encodings()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::VTable;
    use vortex_fastlanes::FoR;

    use super::*;
    use crate::DEFAULT_SCHEMES;

    /// A session with the default schemes and no editions enabled.
    fn session() -> VortexSession {
        vortex_array::array_session()
    }

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn empty_starts_with_no_schemes() {
        let builder = BtrBlocksCompressorBuilder::empty();
        assert!(builder.schemes.is_empty());
    }

    #[test]
    fn from_session_starts_from_registered_schemes_in_order() {
        let builder = BtrBlocksCompressorBuilder::from_session(&session());
        assert_eq!(ids(&builder.schemes), ids(DEFAULT_SCHEMES));
    }

    /// Without enabled editions nothing is permitted, so every scheme is dropped at build.
    #[test]
    fn from_session_permits_only_enabled_editions() {
        let session = session();
        let none = BtrBlocksCompressorBuilder::from_session(&session).permitted_schemes();
        assert!(none.is_empty());
    }

    #[test]
    fn allow_all_keeps_every_scheme() {
        let session = session();
        let schemes = BtrBlocksCompressorBuilder::from_session(&session)
            .allow_all_encodings()
            .permitted_schemes();
        assert_eq!(ids(&schemes), ids(DEFAULT_SCHEMES));
    }

    #[test]
    fn allowed_encodings_filter_schemes_at_build() {
        let session = session();
        let schemes = BtrBlocksCompressorBuilder::from_session(&session)
            .allow_encodings([FoR.id()])
            .permitted_schemes();
        assert_eq!(ids(&schemes), vec![integer::FoRScheme.id()]);
    }

    #[test]
    fn permissions_apply_to_schemes_added_later() {
        let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&integer::FoRScheme);
        assert!(builder.clone().permitted_schemes().is_empty());

        let schemes = builder.allow_encodings([FoR.id()]).permitted_schemes();
        assert_eq!(ids(&schemes), vec![integer::FoRScheme.id()]);
    }

    #[test]
    fn disallowed_encodings_override_allow_all() {
        let session = session();
        let schemes = BtrBlocksCompressorBuilder::from_session(&session)
            .allow_all_encodings()
            .disallow_encodings([FoR.id()])
            .permitted_schemes();
        let ids = ids(&schemes);
        assert!(!ids.contains(&integer::FoRScheme.id()));
        assert!(ids.contains(&integer::BitPackingScheme.id()));
    }

    #[test]
    fn cuda_compatible_excludes_alprd() {
        let builder = BtrBlocksCompressorBuilder::from_session(&session()).only_cuda_compatible();
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
        let builder = BtrBlocksCompressorBuilder::from_session(&session()).only_cuda_compatible();
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
        let builder = BtrBlocksCompressorBuilder::from_session(&session()).only_cuda_compatible();
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
        let builder = BtrBlocksCompressorBuilder::from_session(&session())
            .with_new_scheme(&integer::PcoScheme)
            .with_new_scheme(&float::PcoScheme)
            .only_cuda_compatible();
        for scheme in [integer::PcoScheme.id(), float::PcoScheme.id()] {
            assert!(!builder.schemes.iter().any(|s| s.id() == scheme));
        }
    }
}
