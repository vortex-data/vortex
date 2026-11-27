// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builder for configuring `BtrBlocksCompressor` instances.

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Masked;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_fastlanes::bitpacked_v2_id;
use vortex_fastlanes::for_v2_id;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::CascadingCompressor;
use crate::ChildPolicy;
use crate::CompressionSessionExt;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::allowed_ids::AllowedIds;
use crate::schemes::binary;
use crate::schemes::boolean;
use crate::schemes::fixed_size_list;
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
                // Keep ALP-RD, which has a CUDA decoder; FSST, which has a CUDA decoder and direct
                // Arrow offset-based export; and string dictionaries, which the CUDA dictionary
                // gather decodes into views over the decoded values. Other string fragmentation
                // schemes and binary dictionaries still require unsupported decode paths.
                excluded.extend([
                    boolean::BoolRunEndScheme.id(),
                    boolean::BoolSparseScheme.id(),
                    integer::SparseScheme.id(),
                    integer::IntRLEScheme.id(),
                    float::FloatRLEScheme.id(),
                    float::NullDominatedSparseScheme.id(),
                    string::NullDominatedSparseScheme.id(),
                    fixed_size_list::FixedSizeListSparseScheme.id(),
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

    /// Returns the serialized IDs [`build`](BtrBlocksCompressorBuilder::build) disallows in this
    /// mode, on top of the session's restrictions.
    fn excluded_encodings(self) -> Vec<ArrayId> {
        match self {
            Self::All | Self::Default | Self::Compact => Vec::new(),
            // Multi-part DecimalByteParts arrays, FoR arrays with per-chunk references and
            // BitPacked arrays with per-block bit widths have no CUDA decode kernel.
            Self::Cuda => vec![decimal_byte_parts_v2_id(), for_v2_id(), bitpacked_v2_id()],
        }
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
    child_policy: Option<Arc<dyn ChildPolicy>>,
}

impl BtrBlocksCompressorBuilder {
    /// Creates a builder with every scheme registered in the session's
    /// [`CompressionSession`](crate::CompressionSession), allowing the serialized IDs registered
    /// in the session and permitted by its enabled editions.
    pub fn from_session(session: &VortexSession) -> Self {
        Self {
            schemes: session.compression().schemes(),
            allowed: AllowedIds::from_session(session),
            mode: CompressionMode::Default,
            child_policy: None,
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
            child_policy: None,
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

    /// Sets a policy that decides, child by child, whether to compress the children schemes
    /// register, overriding the schemes' own decisions.
    pub fn with_child_policy(mut self, policy: Arc<dyn ChildPolicy>) -> Self {
        self.child_policy = Some(policy);
        self
    }

    /// Builds the configured [`BtrBlocksCompressor`] from the schemes that its preset does not
    /// exclude and whose produced serialized IDs are all allowed.
    pub fn build(self) -> BtrBlocksCompressor {
        let excluded_ids: HashSet<ArrayId> = self.mode.excluded_encodings().into_iter().collect();
        let allowed = |id: &ArrayId| self.allowed.is_allowed(id) && !excluded_ids.contains(id);
        let compressor = CascadingCompressor::new(self.allowed_schemes_with(&allowed))
            .with_allow_constant(allowed(&Constant.id()))
            .with_allow_masked(allowed(&Masked.id()));
        BtrBlocksCompressor(match self.child_policy {
            Some(policy) => compressor.with_child_policy(policy),
            None => compressor,
        })
    }

    fn allowed_schemes_with(
        &self,
        allowed: &impl Fn(&ArrayId) -> bool,
    ) -> Vec<&'static dyn Scheme> {
        let excluded: HashSet<SchemeId> = self.mode.excluded_schemes().into_iter().collect();
        self.schemes
            .iter()
            .copied()
            .filter(|s| !excluded.contains(&s.id()))
            .map(|s| s.refine(allowed))
            .filter(|s| s.produced_encodings().iter().all(allowed))
            .collect()
    }

    #[cfg(test)]
    fn allowed_schemes(&self) -> Vec<&'static dyn Scheme> {
        let excluded_ids: HashSet<ArrayId> = self.mode.excluded_encodings().into_iter().collect();
        let allowed = |id: &ArrayId| self.allowed.is_allowed(id) && !excluded_ids.contains(id);
        self.allowed_schemes_with(&allowed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::IntoArray;
    use vortex_array::VTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Masked;
    use vortex_array::arrays::Patched;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBin;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_decimal_byte_parts::DecimalByteParts;
    use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
    use vortex_edition::EDITION_DECLARATIONS;
    use vortex_edition::EDITION_FAMILIES;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::declarations::core::CORE_2026_08_3;
    use vortex_edition::declarations::preview::PREVIEW_2026_11_0;
    use vortex_error::VortexResult;
    use vortex_fastlanes::bitpacked_v1_id;
    use vortex_fastlanes::bitpacked_v2_id;
    use vortex_fastlanes::for_v1_id;
    use vortex_fsst::FSST;

    use super::*;
    use crate::CompressionSession;
    use crate::schemes::decimal::DECIMAL_V1;
    use crate::schemes::decimal::DECIMAL_V2;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<CompressionSession>());

    #[rstest]
    #[case::fsst_missing_codes(&string::FSSTScheme, vec![FSST.id()], false)]
    #[case::fsst_with_codes(&string::FSSTScheme, vec![FSST.id(), VarBin.id()], true)]
    #[case::decimal_v1_id(&DECIMAL_V1, vec![decimal_byte_parts_v1_id()], true)]
    #[case::decimal_both_ids(&DECIMAL_V2, vec![decimal_byte_parts_v1_id(), DecimalByteParts.id()], true)]
    #[case::bitpacking_without_constant(
        &integer::BITPACKING_V1,
        vec![bitpacked_v1_id(), Patched.id()],
        false
    )]
    #[case::bitpacking_v1_id(
        &integer::BITPACKING_V1,
        vec![bitpacked_v1_id(), Constant.id(), Patched.id()],
        true
    )]
    #[case::bitpacking_both_ids(
        &integer::BITPACKING_V1,
        vec![bitpacked_v1_id(), bitpacked_v2_id(), Constant.id(), Patched.id()],
        true
    )]
    #[case::bitpacking_v2_only(
        &integer::BITPACKING_V1,
        vec![bitpacked_v2_id(), Constant.id(), Patched.id()],
        false
    )]
    #[case::for_without_bitpacked(&integer::FOR_V1, vec![for_v1_id()], false)]
    #[case::for_with_bitpacked(
        &integer::FOR_V1,
        vec![for_v1_id(), bitpacked_v1_id(), Constant.id(), Patched.id()],
        true
    )]
    #[case::alp_without_constant(
        &float::ALPScheme,
        vec![vortex_alp::ALP.id()],
        false
    )]
    #[case::alp_with_constant(
        &float::ALPScheme,
        vec![vortex_alp::ALP.id(), Constant.id()],
        true
    )]
    #[case::alprd_without_bitpacked(
        &float::ALPRDScheme,
        vec![vortex_alp::ALPRD.id(), Constant.id()],
        false
    )]
    #[case::alprd_without_constant(
        &float::ALPRDScheme,
        vec![vortex_alp::ALPRD.id(), bitpacked_v1_id()],
        false
    )]
    #[case::alprd_with_dependencies(
        &float::ALPRDScheme,
        vec![vortex_alp::ALPRD.id(), bitpacked_v1_id(), Constant.id()],
        true
    )]
    fn test_allowed_schemes(
        #[case] scheme: &'static dyn Scheme,
        #[case] allowed: Vec<ArrayId>,
        #[case] retained: bool,
    ) {
        let mut builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(scheme);
        builder.allowed.editions = Some(allowed.into_iter().collect());

        assert_eq!(!builder.allowed_schemes().is_empty(), retained);
    }

    #[rstest]
    #[case::core(false)]
    #[case::preview(true)]
    fn delta_follows_editions_in_all_writer_presets(#[case] preview: bool) -> VortexResult<()> {
        let session = VortexSession::empty();
        vortex_fastlanes::initialize(&session);
        for family in EDITION_FAMILIES {
            session.editions().declare_family(family)?;
        }
        for declaration in EDITION_DECLARATIONS {
            session.register_edition(declaration)?;
        }
        session.enable_edition(CORE_2026_08_3)?;
        if preview {
            session.enable_edition(PREVIEW_2026_11_0)?;
        }

        let builder = BtrBlocksCompressorBuilder::from_session(&session);
        let delta = integer::DeltaScheme::default().id();
        for preset in [
            builder.clone(),
            builder.clone().with_compact(),
            builder.only_cuda_compatible(),
        ] {
            assert_eq!(
                preset.allowed_schemes().iter().any(|s| s.id() == delta),
                preview
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_keeps_string_dictionaries_but_not_binary_ones() {
        let cuda = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .only_cuda_compatible();
        let has = |id: SchemeId| cuda.allowed_schemes().iter().any(|s| s.id() == id);
        assert!(has(string::StringDictScheme.id()));
        assert!(!has(binary::BinaryDictScheme.id()));
    }

    #[rstest]
    #[case::constant(
        PrimitiveArray::new(buffer![42u64; 1024], Validity::NonNullable).into_array()
    )]
    #[case::all_null(
        PrimitiveArray::new(
            buffer![0u64; 1024],
            Validity::Array(BoolArray::from_iter([false; 1024]).into_array()),
        )
        .into_array()
    )]
    #[case::all_invalid(
        PrimitiveArray::new(buffer![0u64; 1024], Validity::AllInvalid).into_array()
    )]
    #[case::nullable_constant(
        PrimitiveArray::new(
            buffer![42u64; 1024],
            Validity::Array(BoolArray::from_iter((0..1024).map(|i| i % 8 != 0)).into_array()),
        )
        .into_array()
    )]
    #[case::all_valid_bitmap(
        PrimitiveArray::new(
            Buffer::from_iter(0..1024u64),
            Validity::Array(BoolArray::from_iter([true; 1024]).into_array()),
        )
        .into_array()
    )]
    fn test_empty_session_never_emits_constant_or_masked(
        #[case] array: vortex_array::ArrayRef,
    ) -> VortexResult<()> {
        let session = VortexSession::empty();
        let compressor = BtrBlocksCompressor::from_session(&session);
        let mut ctx = session.create_execution_ctx();
        let compressed = compressor.compress(&array, &mut ctx)?;

        assert!(!compressed.is::<Constant>());
        assert!(!compressed.is::<Masked>());
        for slot in compressed.slots().iter().flatten() {
            assert!(!slot.is::<Constant>());
            assert!(!slot.is::<Masked>());
        }
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_constant_without_masked_avoids_masked_for_nullable_constant() -> VortexResult<()> {
        let mut builder = BtrBlocksCompressorBuilder::empty();
        builder.allowed.editions = Some([Constant.id()].into_iter().collect());
        let compressor = builder.build();
        let mut ctx = SESSION.create_execution_ctx();

        let non_null =
            PrimitiveArray::new(buffer![42u64; 1024], Validity::NonNullable).into_array();
        let compressed_non_null = compressor.compress(&non_null, &mut ctx)?;
        assert!(compressed_non_null.is::<Constant>());

        let nullable = PrimitiveArray::new(
            buffer![42u64; 1024],
            Validity::Array(BoolArray::from_iter((0..1024).map(|i| i % 8 != 0)).into_array()),
        )
        .into_array();
        let compressed_nullable = compressor.compress(&nullable, &mut ctx)?;
        assert!(!compressed_nullable.is::<Masked>());
        assert_arrays_eq!(compressed_nullable, nullable, &mut ctx);
        Ok(())
    }
}
