// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.

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

/// Delta, with the minimum compression ratio it needs after its decompression penalty.
///
/// No edition includes `fastlanes.delta` yet, so the session's enabled editions decide whether
/// the compressor may use it. The CUDA preset excludes it.
static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

/// The compression schemes registered on a session, in registration order.
///
/// [`BtrBlocksCompressorBuilder::from_session`](crate::BtrBlocksCompressorBuilder::from_session)
/// starts from these schemes. Registration order is the compressor's tie-break order, so that
/// scheme selection is deterministic.
///
/// [`Default`] registers every built-in scheme, including Delta and the feature-gated Pco and
/// Zstd schemes. The builder's preset decides which of them a compressor uses.
#[derive(Clone, Debug)]
pub struct CompressionSession {
    schemes: Vec<&'static dyn Scheme>,
}

impl Default for CompressionSession {
    fn default() -> Self {
        Self {
            schemes: vec![
                ////////////////////////////////////////////////////////////////////////////////////
                // Integer schemes.
                ////////////////////////////////////////////////////////////////////////////////////
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
                &DELTA_SCHEME,
                #[cfg(feature = "pco")]
                &integer::PcoScheme,
                ////////////////////////////////////////////////////////////////////////////////////
                // Float schemes.
                ////////////////////////////////////////////////////////////////////////////////////
                &float::ALPScheme,
                &float::ALPRDScheme,
                &float::FloatDictScheme,
                &float::NullDominatedSparseScheme,
                &float::FloatRLEScheme,
                #[cfg(feature = "pco")]
                &float::PcoScheme,
                ////////////////////////////////////////////////////////////////////////////////////
                // String schemes.
                ////////////////////////////////////////////////////////////////////////////////////
                &string::StringDictScheme,
                // Both string-fragmentation schemes are registered; the sample-based
                // selector keeps whichever is smaller per column.
                &string::FSSTScheme,
                &string::OnPairScheme,
                &string::NullDominatedSparseScheme,
                #[cfg(feature = "zstd")]
                &string::ZstdScheme,
                ////////////////////////////////////////////////////////////////////////////////////
                // Binary schemes.
                ////////////////////////////////////////////////////////////////////////////////////
                &binary::BinaryDictScheme,
                &binary::VarBinScheme,
                #[cfg(feature = "zstd")]
                &binary::ZstdScheme,
                #[cfg(feature = "zstd")]
                &binary::ZstdBuffersScheme,
                // Decimal schemes.
                &decimal::DecimalScheme,
                // Temporal schemes.
                &temporal::TemporalScheme,
            ],
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

impl CompressionSession {
    /// Registers a compression scheme after the ones already registered.
    ///
    /// This allows encoding crates outside of `vortex-btrblocks` to make their own schemes
    /// available to compressors built from the session.
    ///
    /// # Panics
    ///
    /// Panics if a scheme with the same [`SchemeId`](crate::SchemeId) is already registered.
    pub fn register(&mut self, scheme: &'static dyn Scheme) {
        assert!(
            !self.schemes.iter().any(|s| s.id() == scheme.id()),
            "scheme {:?} is already registered",
            scheme.id(),
        );
        self.schemes.push(scheme);
    }

    /// Returns the registered schemes in registration order.
    pub fn schemes(&self) -> &[&'static dyn Scheme] {
        &self.schemes
    }
}

/// Extension trait for accessing the [`CompressionSession`] of a session.
pub trait CompressionSessionExt: SessionExt {
    /// Returns the session's compression schemes, registering the defaults if absent.
    fn compression(&self) -> SessionGuard<'_, CompressionSession>;

    /// Registers a compression scheme on the session. See [`CompressionSession::register`].
    fn register_scheme(&self, scheme: &'static dyn Scheme);
}

impl<S: SessionExt> CompressionSessionExt for S {
    fn compression(&self) -> SessionGuard<'_, CompressionSession> {
        self.get::<CompressionSession>()
    }

    fn register_scheme(&self, scheme: &'static dyn Scheme) {
        self.get_mut::<CompressionSession>().register(scheme);
    }
}
