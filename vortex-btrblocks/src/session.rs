// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.
//!
//! A session's [`CompressionSession`] holds the schemes available to compressors built from it
//! with [`BtrBlocksCompressor::from_session`](crate::BtrBlocksCompressor::from_session). It
//! starts with the default schemes; [`CompressionSession::compact`] and
//! [`CompressionSession::cuda`] build the other standard registries. Whether a registered scheme
//! may write its encodings is decided by [`BtrBlocksCompressor::from_session`] from the array
//! plugins registered on the session and its enabled editions.

use std::any::Any;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::binary;
use crate::schemes::decimal;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// The default compression schemes, registered by [`CompressionSession::default`].
///
/// This list is order-sensitive: the compressor preserves registration order, so that
/// tie-breaking is deterministic.
const DEFAULT_SCHEMES: &[&dyn Scheme] = &[
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Integer schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
    &integer::FoRScheme,
    // NOTE: ZigZag should precede BitPacking because we don't want negative numbers.
    &integer::ZigZagScheme,
    &integer::BitPackingScheme,
    &integer::SparseScheme,
    &integer::IntDictScheme,
    &integer::RunEndScheme,
    &integer::SequenceScheme,
    &integer::IntRLEScheme,
    // Delta is omitted here: see [`DELTA_SCHEME`].
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Float schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &float::ALPScheme,
    &float::ALPRDScheme,
    &float::FloatDictScheme,
    &float::NullDominatedSparseScheme,
    &float::FloatRLEScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // String schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &string::StringDictScheme,
    // Both string-fragmentation schemes are registered; the sample-based
    // selector keeps whichever is smaller per column.
    &string::FSSTScheme,
    &string::OnPairScheme,
    &string::NullDominatedSparseScheme,
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Binary schemes.
    ////////////////////////////////////////////////////////////////////////////////////////////////
    &binary::BinaryDictScheme,
    &binary::VarBinScheme,
    // Decimal schemes.
    &decimal::DecimalScheme,
    // Temporal schemes.
    &temporal::TemporalScheme,
];

/// The schemes [`CompressionSession::compact`] adds to the defaults: Zstd for strings and binary
/// when the `zstd` feature is on, and Pco for numerics when the `pco` feature is on.
const COMPACT_SCHEMES: &[&dyn Scheme] = &[
    #[cfg(feature = "zstd")]
    &string::ZstdScheme,
    #[cfg(feature = "zstd")]
    &binary::ZstdScheme,
    #[cfg(feature = "pco")]
    &integer::PcoScheme,
    #[cfg(feature = "pco")]
    &float::PcoScheme,
];

/// Delta, kept out of the default schemes because it is slower to decompress than the schemes
/// that would otherwise win. Callers that want it register it and permit `fastlanes.delta`.
///
/// TODO(robert): Return it to the defaults once we have scheme filtering.
pub static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// The compression schemes registered on a session, in registration order.
///
/// Registration order is the compressor's tie-break order between equally good schemes, so
/// sessions that register the same schemes in the same order compress identically.
/// [`Default`] registers the default schemes, [`compact`](Self::compact) and [`cuda`](Self::cuda)
/// their variants, and [`empty`](Self::empty) none.
#[derive(Clone, Debug)]
pub struct CompressionSession {
    /// Registered schemes in registration order.
    schemes: Vec<&'static dyn Scheme>,
}

impl CompressionSession {
    /// A registry with no schemes.
    pub fn empty() -> Self {
        Self {
            schemes: Vec::new(),
        }
    }

    /// The default schemes plus the compact ones: Zstd for strings and binary, and Pco for
    /// numerics, each when its feature is on. They trade decode speed for compression ratio.
    pub fn compact() -> Self {
        let mut this = Self::default();
        for scheme in COMPACT_SCHEMES {
            this.register(*scheme);
        }
        this
    }

    /// The default schemes that CUDA kernels decode, keeping FSST for string compression, plus
    /// Zstd for binary compression when the `zstd` feature is on.
    ///
    /// Both the array-level and the buffer-level Zstd schemes are added. Buffer-level compression
    /// preserves binary arrays' buffer layout for zero-conversion GPU decompression, but belongs
    /// to the opt-in `zstd` edition, so a session's enabled editions decide which of the two a
    /// writer uses. Files written with these schemes may be larger than with the defaults: the
    /// set picks encodings the GPU decodes, not the smallest ones.
    pub fn cuda() -> Self {
        // Keep FSST, which has a CUDA decoder and direct Arrow offset-based export. Other string
        // fragmentation and dictionary schemes still require unsupported decode paths. Delta is
        // not a default either: it has a CUDA decode kernel, but GPU delta decode has not been
        // benchmarked against the schemes it would displace.
        let excluded = [
            integer::SparseScheme.id(),
            integer::IntRLEScheme.id(),
            float::ALPRDScheme.id(),
            float::FloatRLEScheme.id(),
            float::NullDominatedSparseScheme.id(),
            string::NullDominatedSparseScheme.id(),
            string::StringDictScheme.id(),
            binary::BinaryDictScheme.id(),
        ];
        let mut this = Self::empty();
        for scheme in DEFAULT_SCHEMES
            .iter()
            .filter(|scheme| !excluded.contains(&scheme.id()))
        {
            this.register(*scheme);
        }
        #[cfg(feature = "zstd")]
        {
            this.register(&binary::ZstdScheme);
            this.register(&binary::ZstdBuffersScheme);
        }
        this
    }

    /// Registers a scheme.
    ///
    /// Registering a [`SchemeId`](crate::SchemeId) that is already present is a no-op.
    pub fn register(&mut self, scheme: &'static dyn Scheme) {
        if !self.schemes.iter().any(|s| s.id() == scheme.id()) {
            self.schemes.push(scheme);
        }
    }

    /// The registered schemes in registration order.
    pub fn schemes(&self) -> &[&'static dyn Scheme] {
        &self.schemes
    }
}

impl Default for CompressionSession {
    fn default() -> Self {
        Self {
            schemes: DEFAULT_SCHEMES.to_vec(),
        }
    }
}

impl SessionVar for CompressionSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Session access to the compression scheme registry.
pub trait CompressionSessionExt: SessionExt {
    /// Returns the compression scheme registry.
    fn compression(&self) -> SessionGuard<'_, CompressionSession> {
        self.get::<CompressionSession>()
    }

    /// Registers a compression scheme, see [`CompressionSession::register`].
    fn register_scheme(&self, scheme: &'static dyn Scheme) {
        self.get_mut::<CompressionSession>().register(scheme);
    }
}

impl<S: SessionExt> CompressionSessionExt for S {}

#[cfg(test)]
mod tests {
    use vortex_array::array_session;

    use super::*;
    use crate::SchemeId;
    use crate::schemes::float::FloatDictScheme;
    use crate::schemes::integer::IntDictScheme;

    fn ids(schemes: &[&'static dyn Scheme]) -> Vec<SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn default_registers_default_schemes() {
        let session = array_session();
        assert_eq!(ids(session.compression().schemes()), ids(DEFAULT_SCHEMES));
    }

    #[test]
    fn registration_keeps_order_and_is_idempotent() {
        let session = array_session().with_some(CompressionSession::empty());
        session.register_scheme(&IntDictScheme);
        session.register_scheme(&FloatDictScheme);
        session.register_scheme(&IntDictScheme);
        assert_eq!(
            ids(session.compression().schemes()),
            vec![IntDictScheme.id(), FloatDictScheme.id()]
        );
    }

    #[test]
    fn cuda_keeps_fsst_and_drops_string_dict() {
        let cuda = ids(CompressionSession::cuda().schemes());
        assert!(cuda.contains(&string::FSSTScheme.id()));
        assert!(!cuda.contains(&string::StringDictScheme.id()));
    }

    #[test]
    fn compact_extends_the_defaults() {
        let compact = ids(CompressionSession::compact().schemes());
        assert_eq!(&compact[..DEFAULT_SCHEMES.len()], &ids(DEFAULT_SCHEMES)[..]);
        #[cfg(feature = "zstd")]
        assert!(compact.contains(&string::ZstdScheme.id()));
    }
}
