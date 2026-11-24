// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Overriding the compressor's decisions: which scheme each array gets, and which children are
//! compressed.

use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;

use crate::scheme::ChildDecision;
use crate::scheme::ChildRequest;
use crate::scheme::CompressorContext;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// How the compressor chooses the scheme for one array, as a [`CompressionPolicy`] directs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemeSelection {
    /// Estimate every eligible scheme and compress with the winner, as the compressor does
    /// without a policy.
    Estimate,
    /// Leave the array in its canonical encoding.
    Canonical,
    /// Compress with this scheme, without estimating.
    ///
    /// If the scheme is not eligible for the array, fails on it, or does not make it smaller, the
    /// compressor estimates as for [`Estimate`](Self::Estimate).
    Scheme(SchemeId),
    /// Compress the array in full with each of these schemes and keep the output the policy
    /// [ranks](CompressionPolicy::rank) lowest. An empty list tries every eligible scheme.
    ///
    /// Schemes that are not eligible, fail, or do not make the array smaller are skipped. If none
    /// is left, the array stays canonical.
    Best(Vec<SchemeId>),
    /// Estimate only these schemes and compress with the winner.
    ///
    /// Schemes that are not eligible are ignored. If none of them is eligible, the compressor
    /// estimates as for [`Estimate`](Self::Estimate).
    EstimateAmong(Vec<SchemeId>),
}

/// An array the compressor is about to choose a scheme for, as seen by a [`CompressionPolicy`].
#[derive(Debug)]
pub struct SiteRequest<'a> {
    /// The canonical array.
    pub array: &'a ArrayRef,
    /// The schemes that match the array and that the exclusion rules allow, in registration
    /// order.
    pub eligible: &'a [SchemeId],
    /// The compression site. Its [`cascade_history`](CompressorContext::cascade_history) is the
    /// path from the root array to this one.
    pub ctx: &'a CompressorContext,
}

/// The scheme the compressor applied to an array, as reported to a [`CompressionPolicy`].
#[derive(Debug)]
pub struct SiteOutcome<'a> {
    /// The applied scheme, or `None` when the array was left canonical.
    pub scheme: Option<SchemeId>,
    /// The array the site produced.
    pub array: &'a ArrayRef,
    /// The compression site.
    pub ctx: &'a CompressorContext,
}

/// Lets a caller override the compressor's decisions, for search, plans and learned selectors.
///
/// Every method has a default that keeps the compressor's own behavior, so a policy implements
/// only what it changes.
///
/// The compressor consults the policy at every compression site of the full array: the root and
/// each cascaded child. It does not consult it while compressing samples for estimates. Empty,
/// all-null and constant arrays are handled before the policy is asked.
pub trait CompressionPolicy: Debug + Send + Sync {
    /// Decides whether a child a scheme registered is compressed or kept as it is.
    ///
    /// A policy can keep a child a scheme asked to compress, or compress a child a scheme asked
    /// to keep. Compressing a kept child is the policy's responsibility: the scheme may rely on it
    /// staying canonical. The cascade budget still applies: once a chain has used up
    /// [`MAX_CASCADE`] levels, its children are kept whatever the policy decides.
    ///
    /// [`MAX_CASCADE`]: crate::scheme::MAX_CASCADE
    fn decide_child(&self, request: &ChildRequest<'_>) -> ChildDecision {
        request.requested
    }

    /// Chooses how the compressor selects the scheme for an array.
    fn select_scheme(&self, _request: &SiteRequest<'_>) -> SchemeSelection {
        SchemeSelection::Estimate
    }

    /// Chooses how the compressor selects the scheme for an array, with access to the statistics
    /// the eligible schemes requested.
    ///
    /// Statistics read through `data` are cached and shared with the schemes, so a policy that
    /// reads them adds no extra pass over the array. Defaults to
    /// [`select_scheme`](Self::select_scheme).
    fn select_scheme_with_stats(
        &self,
        request: &SiteRequest<'_>,
        _data: &ArrayAndStats,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeSelection {
        self.select_scheme(request)
    }

    /// Ranks a candidate output of [`SchemeSelection::Best`]; the lowest wins.
    ///
    /// Defaults to the array's size in bytes.
    fn rank(&self, candidate: &ArrayRef, _ctx: &CompressorContext) -> f64 {
        candidate.nbytes() as f64
    }

    /// Observes the scheme the compressor applied to an array.
    ///
    /// Called once per compression site, after its children were compressed. Under
    /// [`SchemeSelection::Best`], the children of every tried scheme are reported, each under its
    /// own cascade history, so a recorder keeps only those below the schemes their parents
    /// applied.
    fn observe(&self, _outcome: &SiteOutcome<'_>) {}

    /// Observes one candidate tried under [`SchemeSelection::Best`], winner or not, with its full
    /// output.
    ///
    /// Called once per candidate that succeeded and made the array smaller, before
    /// [`observe`](Self::observe) reports the site's winner. The candidate's children were
    /// observed first, under cascade histories that end in its scheme.
    fn observe_candidate(&self, _candidate: &SiteOutcome<'_>) {}
}
