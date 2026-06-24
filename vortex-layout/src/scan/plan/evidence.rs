// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Predicate evidence: coverage-bearing answers for prepared predicates.
//!
//! A scan predicate is answered at runtime by *evidence fragments*:
//! row ranges paired with what a producer proves about the
//! predicate over them. A whole-morsel verdict is the degenerate case of one
//! fragment covering the morsel; finer coverage is first-class, so a zone map
//! can prove interior zones while leaving edge rows unknown, and an index can
//! return sparse row masks without forcing the whole morsel down the same path.
//!
//! Exactness is explicit in the returned evidence kind.
//! [`PredicateEvidenceKind::ExactMask`] proves both selected and rejected
//! rows — the source may suppress residual evaluation for the covered
//! range. [`PredicateEvidenceKind::CandidateMask`] proves only that
//! masked-out rows are rejected; masked-in rows must still run the
//! residual predicate. Approximate producers must return candidate
//! evidence directly.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

/// Identifies one predicate of a scan. Stable for the lifetime
/// of the expanded scan: producers and the source combine evidence by
/// predicate id (never by expression text), so rewritten or derived
/// predicate forms stay tied to their original predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PredicateId(u32);

impl PredicateId {
    /// The id of the `idx`-th predicate.
    pub fn new(idx: u32) -> Self {
        Self(idx)
    }

    /// This id as an index into the scan's predicate list.
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for PredicateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// Distinguishes successive values of a dynamic predicate (a runtime
/// boundary that tightens between morsels). Static predicates stay at
/// version zero. Evidence only combines within one (id, version) pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PredicateVersion(u64);

impl PredicateVersion {
    /// The version for static predicates.
    pub const STATIC: Self = Self(0);

    /// A dynamic predicate's version, from its boundary slot.
    pub fn new(version: u64) -> Self {
        Self(version)
    }
}

/// Exactness metadata for producers that need to degrade an evidence kind
/// before returning it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exactness {
    /// Evidence may be exact: `AllTrue`, `AllFalse`, and `ExactMask` are
    /// accepted as returned.
    Exact,
    /// Evidence is at most a candidate: masked-out rows are rejected,
    /// masked-in rows must still run the residual predicate. `ExactMask`
    /// degrades to `CandidateMask` and `AllTrue` to `Unknown`; `AllFalse`
    /// stays, since rejecting rows is within a candidate's authority.
    Candidate,
}

impl Exactness {
    /// The stronger of two exactness values.
    #[allow(dead_code)]
    pub(crate) fn max(self, other: Self) -> Self {
        match (self, other) {
            (Self::Candidate, Self::Candidate) => Self::Candidate,
            _ => Self::Exact,
        }
    }
}

/// What a fragment proves about a predicate over its row range.
#[derive(Clone, Debug)]
pub enum PredicateEvidenceKind {
    /// The predicate is false for every row in the range.
    AllFalse,
    /// The predicate is true for every row in the range.
    AllTrue,
    /// Exact per-row verdicts: set rows are true, unset rows are false.
    /// Residual evaluation is unnecessary for the covered range.
    ExactMask(Mask),
    /// Unset rows are proven false; set rows are only candidates and must
    /// still run the residual predicate. Approximate indexes must use
    /// this kind.
    CandidateMask(Mask),
    /// The producer proves nothing about the range.
    Unknown,
}

impl PredicateEvidenceKind {
    /// This kind degraded to a producer's advertised exactness.
    #[allow(dead_code)]
    pub(crate) fn cap(self, ceiling: Exactness) -> Self {
        match ceiling {
            Exactness::Exact => self,
            Exactness::Candidate => match self {
                Self::AllTrue => Self::Unknown,
                Self::ExactMask(mask) => Self::CandidateMask(mask),
                other => other,
            },
        }
    }
}

/// One producer's answer for one predicate over one row range, in the
/// producer's row coordinates (for column producers these coincide with
/// file row coordinates).
#[derive(Clone, Debug)]
pub struct EvidenceFragment {
    /// The rows this fragment covers.
    pub rows: Range<u64>,
    /// What is proven over them.
    pub kind: PredicateEvidenceKind,
}

impl EvidenceFragment {
    /// A fragment proving `kind` over `rows`.
    pub fn new(rows: Range<u64>, kind: PredicateEvidenceKind) -> Self {
        Self { rows, kind }
    }
}

/// Accumulated evidence for one predicate over one morsel: fragments fold
/// into two morsel-local masks.
///
/// - `maybe`: rows that may still satisfy the predicate. Starts all-true;
///   every proof of falseness clears bits.
/// - `proven`: rows whose verdict is exactly known (true *or* false), so
///   residual evaluation cannot change it. Starts all-false.
///
/// The invariant `!maybe ⊆ proven` holds throughout: a row is only
/// removed from `maybe` by evidence that proves it false.
pub struct PredicateEvidence {
    id: PredicateId,
    version: PredicateVersion,
    /// The morsel's row range in file coordinates.
    range: Range<u64>,
    maybe: Mask,
    proven: Mask,
}

impl PredicateEvidence {
    /// Fresh evidence (nothing proven) for one predicate over the morsel
    /// `range`.
    pub fn new(
        id: PredicateId,
        version: PredicateVersion,
        range: Range<u64>,
    ) -> VortexResult<Self> {
        let len = range_len(&range)?;
        Ok(Self {
            id,
            version,
            range,
            maybe: Mask::new_true(len),
            proven: Mask::new_false(len),
        })
    }

    /// The predicate this evidence answers.
    pub fn id(&self) -> PredicateId {
        self.id
    }

    /// Rows that may still satisfy the predicate (morsel-local).
    pub fn maybe(&self) -> &Mask {
        &self.maybe
    }

    /// Rows whose verdict residual evaluation may not change
    /// (morsel-local).
    pub fn proven(&self) -> &Mask {
        &self.proven
    }

    /// Rows that may satisfy the predicate but are not exactly proven:
    /// the rows the residual predicate must evaluate.
    pub fn unproven(&self) -> Mask {
        // The common whole-morsel verdicts skip the bit traversal: with
        // nothing proven everything in `maybe` is residual, and with
        // everything proven nothing is.
        if self.proven.all_false() {
            return self.maybe.clone();
        }
        if self.proven.all_true() {
            return Mask::new_false(self.maybe.len());
        }
        self.maybe.clone().bitand_not(&self.proven)
    }

    /// Whether no row of the morsel can satisfy the predicate.
    pub fn all_false(&self) -> bool {
        self.maybe.all_false()
    }

    /// Fold one fragment in. Fragments outside the morsel range are
    /// clipped (wholly disjoint fragments are ignored); fragment masks
    /// must match their declared row range.
    pub fn absorb(&mut self, fragment: EvidenceFragment) -> VortexResult<()> {
        let span = fragment.rows.start.max(self.range.start)..fragment.rows.end.min(self.range.end);
        if span.start >= span.end {
            return Ok(());
        }
        // The covered span in morsel-local coordinates.
        let local = usize::try_from(span.start - self.range.start)
            .map_err(|_| vortex_err!("morsel exceeds usize"))?
            ..usize::try_from(span.end - self.range.start)
                .map_err(|_| vortex_err!("morsel exceeds usize"))?;
        let len = self.maybe.len();
        // Fragments covering the whole morsel — the dominant case when a
        // zone run spans it — combine without building placement masks.
        let whole = local.start == 0 && local.end == len;
        match fragment.kind {
            PredicateEvidenceKind::Unknown => {}
            PredicateEvidenceKind::AllFalse if whole => {
                self.maybe = Mask::new_false(len);
                self.proven = Mask::new_true(len);
            }
            PredicateEvidenceKind::AllFalse => {
                self.maybe = &self.maybe & &constrain(len, &local, None)?;
                self.proven = &self.proven | &prove(len, &local, None)?;
            }
            PredicateEvidenceKind::AllTrue if whole => {
                self.proven = Mask::new_true(len);
            }
            PredicateEvidenceKind::AllTrue => {
                self.proven = &self.proven | &prove(len, &local, None)?;
            }
            PredicateEvidenceKind::ExactMask(mask) if whole => {
                let mask = clip_mask(mask, &fragment.rows, &span)?;
                self.maybe = &self.maybe & &mask;
                self.proven = Mask::new_true(len);
            }
            PredicateEvidenceKind::ExactMask(mask) => {
                let mask = clip_mask(mask, &fragment.rows, &span)?;
                self.maybe = &self.maybe & &constrain(len, &local, Some(&mask))?;
                self.proven = &self.proven | &prove(len, &local, None)?;
            }
            PredicateEvidenceKind::CandidateMask(mask) if whole => {
                let mask = clip_mask(mask, &fragment.rows, &span)?;
                self.maybe = &self.maybe & &mask;
                self.proven = &self.proven | &!mask;
            }
            PredicateEvidenceKind::CandidateMask(mask) => {
                let mask = clip_mask(mask, &fragment.rows, &span)?;
                self.maybe = &self.maybe & &constrain(len, &local, Some(&mask))?;
                // Only the masked-out rows are proven (false); masked-in
                // rows remain candidates.
                let rejected = !mask;
                self.proven = &self.proven | &prove(len, &local, Some(&rejected))?;
            }
        }
        Ok(())
    }

    /// The version this evidence was requested at.
    pub fn version(&self) -> PredicateVersion {
        self.version
    }
}

/// A morsel-length AND-constraint: `mask` (or all-false, proving the span
/// rejected) inside `span`, `true` — the AND identity — outside it.
fn constrain(len: usize, span: &Range<usize>, mask: Option<&Mask>) -> VortexResult<Mask> {
    placed(len, span, mask, false, true)
}

/// A morsel-length proof: `mask` (or all-true, proving the whole span)
/// inside `span`, `false` — the OR identity — outside it.
fn prove(len: usize, span: &Range<usize>, mask: Option<&Mask>) -> VortexResult<Mask> {
    placed(len, span, mask, true, false)
}

/// `mask` (or `fill`) placed at `span`, `outside` everywhere else.
fn placed(
    len: usize,
    span: &Range<usize>,
    mask: Option<&Mask>,
    fill: bool,
    outside: bool,
) -> VortexResult<Mask> {
    let span_len = span.end - span.start;
    let inner = match mask {
        Some(mask) => {
            if mask.len() != span_len {
                vortex_bail!(
                    "evidence mask length {} does not match its row range length {span_len}",
                    mask.len()
                );
            }
            mask.clone()
        }
        None => Mask::new(span_len, fill),
    };
    let lead = Mask::new(span.start, outside);
    let tail = Mask::new(len - span.end, outside);
    Mask::concat([&lead, &inner, &tail].into_iter())
}

/// Slice a fragment's mask down to the morsel-clipped part of its range.
fn clip_mask(mask: Mask, rows: &Range<u64>, span: &Range<u64>) -> VortexResult<Mask> {
    let full = range_len(rows)?;
    if mask.len() != full {
        vortex_bail!(
            "evidence mask length {} does not match its row range {rows:?}",
            mask.len()
        );
    }
    if span == rows {
        return Ok(mask);
    }
    let start = usize::try_from(span.start - rows.start)
        .map_err(|_| vortex_err!("evidence fragment exceeds usize"))?;
    let end = usize::try_from(span.end - rows.start)
        .map_err(|_| vortex_err!("evidence fragment exceeds usize"))?;
    Ok(mask.slice(start..end))
}

fn range_len(range: &Range<u64>) -> VortexResult<usize> {
    usize::try_from(range.end.saturating_sub(range.start))
        .map_err(|_| vortex_err!("row range {range:?} exceeds usize"))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use super::*;

    fn evidence(range: Range<u64>) -> PredicateEvidence {
        PredicateEvidence::new(PredicateId::new(0), PredicateVersion::STATIC, range)
            .vortex_expect("fresh evidence")
    }

    /// Nothing absorbed: everything is a candidate, nothing is proven.
    #[test]
    fn fresh_evidence_proves_nothing() {
        let acc = evidence(100..200);
        assert!(acc.maybe().all_true());
        assert!(acc.proven().all_false());
        assert!(acc.unproven().all_true());
        assert!(!acc.all_false());
    }

    /// A whole-range AllFalse fragment kills the morsel.
    #[test]
    fn all_false_whole_range() -> VortexResult<()> {
        let mut acc = evidence(100..200);
        acc.absorb(EvidenceFragment::new(
            100..200,
            PredicateEvidenceKind::AllFalse,
        ))?;
        assert!(acc.all_false());
        assert!(acc.proven().all_true());
        assert!(acc.unproven().all_false());
        Ok(())
    }

    /// Partial coverage: an interior AllFalse zone clears only its span
    /// and leaves the edges unproven.
    #[test]
    fn partial_all_false_leaves_edges_unproven() -> VortexResult<()> {
        let mut acc = evidence(100..200);
        acc.absorb(EvidenceFragment::new(
            120..150,
            PredicateEvidenceKind::AllFalse,
        ))?;
        assert!(!acc.all_false());
        assert_eq!(acc.maybe().true_count(), 70);
        assert_eq!(acc.proven().true_count(), 30);
        assert!(!acc.maybe().value(20)); // row 120
        assert!(acc.maybe().value(50)); // row 150
        assert_eq!(acc.unproven().true_count(), 70);
        Ok(())
    }

    /// AllTrue proves rows without shrinking the candidate set.
    #[test]
    fn all_true_proves_without_filtering() -> VortexResult<()> {
        let mut acc = evidence(0..100);
        acc.absorb(EvidenceFragment::new(0..40, PredicateEvidenceKind::AllTrue))?;
        assert!(acc.maybe().all_true());
        assert_eq!(acc.proven().true_count(), 40);
        assert_eq!(acc.unproven().true_count(), 60);
        Ok(())
    }

    /// An exact mask proves its whole span: selected rows survive,
    /// rejected rows leave, and no residual evaluation remains there.
    #[test]
    fn exact_mask_proves_whole_span() -> VortexResult<()> {
        let mut acc = evidence(0..100);
        acc.absorb(EvidenceFragment::new(
            10..20,
            PredicateEvidenceKind::ExactMask(Mask::from_indices(10, [2, 5])),
        ))?;
        assert_eq!(acc.maybe().true_count(), 92);
        assert!(acc.maybe().value(12));
        assert!(acc.maybe().value(15));
        assert!(!acc.maybe().value(13));
        assert_eq!(acc.proven().true_count(), 10);
        // The two surviving rows are proven, not residual candidates.
        assert!(!acc.unproven().value(12));
        Ok(())
    }

    /// A candidate mask rejects masked-out rows but keeps masked-in rows
    /// residual.
    #[test]
    fn candidate_mask_keeps_residual() -> VortexResult<()> {
        let mut acc = evidence(0..100);
        acc.absorb(EvidenceFragment::new(
            10..20,
            PredicateEvidenceKind::CandidateMask(Mask::from_indices(10, [2, 5])),
        ))?;
        assert_eq!(acc.maybe().true_count(), 92);
        // Rejected rows are proven false; candidates are not proven.
        assert_eq!(acc.proven().true_count(), 8);
        assert!(acc.unproven().value(12));
        assert!(acc.unproven().value(15));
        assert!(!acc.unproven().value(13));
        Ok(())
    }

    /// Fragments combine: evidence from several producers intersects
    /// candidates and unions proofs.
    #[test]
    fn fragments_combine_across_producers() -> VortexResult<()> {
        let mut acc = evidence(0..100);
        // A zone map proves rows 0..50 all-false.
        acc.absorb(EvidenceFragment::new(
            0..50,
            PredicateEvidenceKind::AllFalse,
        ))?;
        // An index proves rows 40..100 exactly: only rows 60 and 70 match.
        acc.absorb(EvidenceFragment::new(
            40..100,
            PredicateEvidenceKind::ExactMask(Mask::from_indices(60, [20, 30])),
        ))?;
        assert_eq!(acc.maybe().true_count(), 2);
        assert!(acc.maybe().value(60));
        assert!(acc.maybe().value(70));
        assert!(acc.proven().all_true());
        assert!(acc.unproven().all_false());
        Ok(())
    }

    /// Fragments are clipped to the morsel range, masks included.
    #[test]
    fn fragments_clip_to_range() -> VortexResult<()> {
        let mut acc = evidence(100..200);
        // Covers 50..150 in file coordinates; only 100..150 is in range.
        acc.absorb(EvidenceFragment::new(
            50..150,
            PredicateEvidenceKind::ExactMask(Mask::from_indices(100, [40, 60])),
        ))?;
        // Index 40 falls before the morsel; index 60 is row 110.
        assert_eq!(acc.maybe().true_count(), 51);
        assert!(acc.maybe().value(10));
        assert_eq!(acc.proven().true_count(), 50);
        // Wholly disjoint fragments are ignored.
        acc.absorb(EvidenceFragment::new(
            0..100,
            PredicateEvidenceKind::AllFalse,
        ))?;
        assert_eq!(acc.maybe().true_count(), 51);
        Ok(())
    }

    /// A mask whose length does not match its declared range is an error.
    #[test]
    fn mismatched_mask_length_is_an_error() {
        let mut acc = evidence(0..100);
        let result = acc.absorb(EvidenceFragment::new(
            0..50,
            PredicateEvidenceKind::ExactMask(Mask::new_true(10)),
        ));
        assert!(result.is_err());
    }

    /// Candidate ceilings degrade exact evidence kinds.
    #[test]
    fn exactness_ceiling_caps_kinds() {
        let exact = PredicateEvidenceKind::ExactMask(Mask::new_true(4));
        assert!(matches!(
            exact.cap(Exactness::Candidate),
            PredicateEvidenceKind::CandidateMask(_)
        ));
        assert!(matches!(
            PredicateEvidenceKind::AllTrue.cap(Exactness::Candidate),
            PredicateEvidenceKind::Unknown
        ));
        // Rejection stays within a candidate's authority.
        assert!(matches!(
            PredicateEvidenceKind::AllFalse.cap(Exactness::Candidate),
            PredicateEvidenceKind::AllFalse
        ));
    }
}
