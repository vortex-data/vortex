// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session registry of compression schemes.

use std::any::Any;
use std::sync::Arc;

use parking_lot::RwLock;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::schemes::binary;
use crate::schemes::boolean;
use crate::schemes::decimal;
use crate::schemes::fixed_size_list;
use crate::schemes::float;
use crate::schemes::integer;
use crate::schemes::string;
use crate::schemes::temporal;

/// Delta, with the minimum compression ratio it needs after its decompression penalty.
///
/// `fastlanes.delta` is in the `preview` edition, not `core`, so the session's enabled editions
/// decide whether the compressor may use it.
static DELTA_SCHEME: integer::DeltaScheme = integer::DeltaScheme::new(1.25);

static ONPAIR_SCHEME: string::OnPairScheme =
    string::OnPairScheme::new(DEFAULT_CONFIG.max_dict_bits);

type SchemeRegistration = (SchemeId, &'static dyn Scheme);
type SchemeRegistry = Arc<RwLock<Vec<SchemeRegistration>>>;

/// The compression schemes registered on a session, in registration order.
///
/// [`BtrBlocksCompressorBuilder::from_session`](crate::BtrBlocksCompressorBuilder::from_session)
/// starts from these schemes. Registration order is the compressor's tie-break order, so that
/// scheme selection is deterministic.
///
/// Clones share the same synchronized registry. [`schemes`](Self::schemes) returns an owned
/// snapshot, so later registrations do not change an existing snapshot.
///
/// [`Default`] registers every built-in scheme, including Delta and the feature-gated Pco and
/// Zstd schemes. The builder's preset decides which of them a compressor uses.
#[derive(Clone, Debug)]
pub struct CompressionSession {
    schemes: SchemeRegistry,
}

impl Default for CompressionSession {
    fn default() -> Self {
        let schemes: Vec<&'static dyn Scheme> = vec![
            &boolean::BoolRunEndScheme,
            &boolean::BoolSparseScheme,
            ////////////////////////////////////////////////////////////////////////////////////
            // Integer schemes.
            ////////////////////////////////////////////////////////////////////////////////////
            // NOTE: FoR must precede BitPacking to avoid unnecessary patches.
            // `refine` switches FOR_V1 to FOR_V2 where `fastlanes.for.v2` and `fastlanes.for.v3` are allowed.
            &integer::FOR_V1,
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
            &ONPAIR_SCHEME,
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
            // Decimal schemes. `refine` switches to v2 where the v2 ID is allowed.
            // Byte parts are the default, which the dictionary's estimate has to beat.
            &decimal::DecimalDictScheme,
            &decimal::DECIMAL_V1,
            // Temporal schemes.
            &temporal::TemporalScheme,
            // Fixed-size list schemes.
            &fixed_size_list::FixedSizeListSparseScheme,
        ];
        let schemes: Vec<SchemeRegistration> = schemes
            .into_iter()
            .map(|scheme| (scheme.id(), scheme))
            .collect();

        Self {
            schemes: Arc::new(RwLock::new(schemes)),
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
    /// Panics if a scheme with the same [`SchemeId`] is already registered.
    pub fn register(&self, scheme: &'static dyn Scheme) {
        let id = scheme.id();
        let mut schemes = self.schemes.write();
        assert!(
            !schemes
                .iter()
                .any(|(registered_id, _)| *registered_id == id),
            "scheme {:?} is already registered",
            id,
        );
        schemes.push((id, scheme));
    }

    /// Returns a stable snapshot of the registered schemes in registration order.
    pub fn schemes(&self) -> Vec<&'static dyn Scheme> {
        self.schemes
            .read()
            .iter()
            .map(|(_, scheme)| *scheme)
            .collect()
    }
}

/// Extension trait for accessing the [`CompressionSession`] of a session.
pub trait CompressionSessionExt: SessionExt {
    /// Returns the session's compression schemes, registering the defaults if absent.
    fn compression(&self) -> SessionGuard<'_, CompressionSession>;

    /// Registers a compression scheme on the session. Registration is synchronized and shared
    /// across cloned [`CompressionSession`]s. See [`CompressionSession::register`].
    fn register_scheme(&self, scheme: &'static dyn Scheme);
}

impl<S: SessionExt> CompressionSessionExt for S {
    fn compression(&self) -> SessionGuard<'_, CompressionSession> {
        self.get::<CompressionSession>()
    }

    fn register_scheme(&self, scheme: &'static dyn Scheme) {
        self.get::<CompressionSession>().register(scheme);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_compressor::CascadingCompressor;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::CompressionSession;
    use super::CompressionSessionExt;
    use crate::ArrayAndStats;
    use crate::CompressorContext;
    use crate::Scheme;
    use crate::SchemeExt;
    use crate::SchemeId;

    #[derive(Debug)]
    struct TestScheme(&'static str);

    impl Scheme for TestScheme {
        fn scheme_name(&self) -> &'static str {
            self.0
        }

        fn matches(&self, _canonical: &Canonical) -> bool {
            false
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unreachable!()
        }
    }

    static FIRST_SCHEME: TestScheme = TestScheme("vortex.test.registry.first");
    static SECOND_SCHEME: TestScheme = TestScheme("vortex.test.registry.second");

    fn scheme_ids(schemes: &[&'static dyn Scheme]) -> Vec<SchemeId> {
        schemes.iter().map(|scheme| scheme.id()).collect()
    }

    #[test]
    fn test_clones_share_registrations_and_snapshots_remain_stable() {
        let session = CompressionSession::default();
        let snapshot = session.schemes();
        let snapshot_ids = scheme_ids(&snapshot);

        session.clone().register(&FIRST_SCHEME);

        assert_eq!(scheme_ids(&snapshot), snapshot_ids);
        let schemes = session.schemes();
        assert_eq!(schemes.last().unwrap().id(), FIRST_SCHEME.id());
    }

    #[test]
    fn test_concurrent_session_registrations_keep_both_schemes_and_default_order() {
        let default_ids = scheme_ids(&CompressionSession::default().schemes());
        let session = VortexSession::empty();
        let barrier = Arc::new(Barrier::new(3));

        let first_session = session.clone();
        let first_barrier = Arc::clone(&barrier);
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_session.register_scheme(&FIRST_SCHEME);
        });

        let second_session = session.clone();
        let second_barrier = Arc::clone(&barrier);
        let second = thread::spawn(move || {
            second_barrier.wait();
            second_session.register_scheme(&SECOND_SCHEME);
        });

        barrier.wait();
        first.join().unwrap();
        second.join().unwrap();

        let schemes = session.compression().schemes();
        let ids = scheme_ids(&schemes);
        assert_eq!(&ids[..default_ids.len()], default_ids.as_slice());
        assert!(ids.contains(&FIRST_SCHEME.id()));
        assert!(ids.contains(&SECOND_SCHEME.id()));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_duplicate_scheme_registration_panics() {
        let session = CompressionSession::default();

        session.register(&FIRST_SCHEME);
        session.register(&FIRST_SCHEME);
    }
}
