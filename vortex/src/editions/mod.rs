// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Vortex edition declarations.
//!
//! [`vortex_edition`] provides the types, the [`EditionSession`] session variable, and the
//! test harness; the actual declarations live here, one module per edition
//! (`editions::<family>::<date>`), and are seeded into the default session by
//! [`register_default_editions`].
//!
//! Each edition module declares the edition together with the encodings that join the
//! family at it; members of earlier editions are inherited and never restated. Correctness
//! is enforced by unit tests: every edition module calls
//! [`vortex_edition::test_harness::validate_edition`] once from its `#[cfg(test)]` module,
//! and the computed set of a frozen edition is pinned by a golden test, so any change to
//! these declarations that alters a frozen set fails CI. New encodings are staged into the
//! newest draft edition.

pub mod core;

use vortex_edition::EditionDeclaration;
pub use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

pub use self::core::CORE_2026_01_0;
pub use self::core::CORE_2026_07_0;

/// The Vortex editions, each declared together with the encodings it adds.
pub static DEFAULT_DECLARATIONS: &[&EditionDeclaration] =
    &[&core::v2026_01::DECLARATION, &core::v2026_07::DECLARATION];

/// Register the Vortex edition declarations with the session's [`EditionSession`].
pub fn register_default_editions(session: &VortexSession) {
    let editions = session.editions();
    for declaration in DEFAULT_DECLARATIONS {
        editions
            .declare(declaration)
            .map_err(|e| vortex_err!("{e}"))
            .vortex_expect("default edition declarations are valid");
    }
}

/// An [`EditionSession`] seeded with the Vortex edition declarations, for tests.
#[cfg(test)]
pub(crate) fn edition_session() -> EditionSession {
    let session = VortexSession::empty();
    register_default_editions(&session);
    session.editions().clone()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vortex_edition::EditionId;
    use vortex_edition::EditionInclusion;
    use vortex_edition::EditionSessionExt;
    use vortex_session::VortexSession;

    use super::CORE_2026_01_0;
    use super::CORE_2026_07_0;
    use super::DEFAULT_DECLARATIONS;
    use super::edition_session;

    /// The intended encoding set of `core2026.01.0`: the canonical encodings.
    ///
    /// While an edition is a draft this is a deliberate change-detector; once the edition
    /// is frozen (its `min_vortex_version` recorded), the set must never change again —
    /// revert the change or stage it into the next draft edition instead.
    const FROZEN_CORE_2026_01_0: &[&str] = &[
        "vortex.bool",
        "vortex.decimal",
        "vortex.ext",
        "vortex.fixed_size_list",
        "vortex.list",
        "vortex.listview",
        "vortex.null",
        "vortex.primitive",
        "vortex.struct",
        "vortex.varbin",
        "vortex.varbinview",
        "vortex.variant",
    ];

    /// The intended encoding set of `core2026.07.0`: everything in `core2026.01.0` (by
    /// inheritance) plus the container and compressed encodings.
    const FROZEN_CORE_2026_07_0: &[&str] = &[
        "fastlanes.bitpacked",
        "fastlanes.delta",
        "fastlanes.for",
        "fastlanes.rle",
        "vortex.alp",
        "vortex.alprd",
        "vortex.bool",
        "vortex.bytebool",
        "vortex.chunked",
        "vortex.constant",
        "vortex.datetimeparts",
        "vortex.decimal",
        "vortex.decimal_byte_parts",
        "vortex.dict",
        "vortex.ext",
        "vortex.fixed_size_list",
        "vortex.fsst",
        "vortex.list",
        "vortex.listview",
        "vortex.masked",
        "vortex.null",
        "vortex.pco",
        "vortex.primitive",
        "vortex.runend",
        "vortex.sequence",
        "vortex.sparse",
        "vortex.struct",
        "vortex.varbin",
        "vortex.varbinview",
        "vortex.variant",
        "vortex.zigzag",
        "vortex.zstd",
    ];

    #[test]
    fn frozen_core_2026_01_0() {
        let editions = edition_session();
        let members = editions.encodings_in(&CORE_2026_01_0);
        let ids: Vec<&str> = members.iter().map(|i| i.encoding_id.as_str()).collect();
        assert_eq!(ids, FROZEN_CORE_2026_01_0);
    }

    #[test]
    fn frozen_core_2026_07_0() {
        let editions = edition_session();
        let members = editions.encodings_in(&CORE_2026_07_0);
        let ids: Vec<&str> = members.iter().map(|i| i.encoding_id.as_str()).collect();
        assert_eq!(ids, FROZEN_CORE_2026_07_0);
    }

    #[test]
    fn membership_is_transitive() {
        let editions = edition_session();

        // Everything in the first edition is in the second by inheritance, with its
        // `since` still recording the edition it actually joined in.
        let second = editions.encodings_in(&CORE_2026_07_0);
        for canonical in FROZEN_CORE_2026_01_0 {
            let member = second
                .iter()
                .find(|i| i.encoding_id.as_str() == *canonical)
                .unwrap_or_else(|| panic!("{canonical} missing from core2026.07.0"));
            assert_eq!(member.since, CORE_2026_01_0);
        }

        // The second edition's delta is exactly the members declared at it.
        let added = second.iter().filter(|i| i.since == CORE_2026_07_0).count();
        assert_eq!(
            added,
            FROZEN_CORE_2026_07_0.len() - FROZEN_CORE_2026_01_0.len()
        );

        // Inheritance never flows backwards.
        let first = editions.encodings_in(&CORE_2026_01_0);
        assert!(first.iter().all(|i| i.since == CORE_2026_01_0));
    }

    /// Validates the declaration blocks directly, before any session exists: every edition
    /// id is well-formed and declared once, and every added encoding is well-formed and
    /// named only once across all blocks — one membership interval per encoding, which
    /// also means at most one per family. (An inclusion referencing an undeclared edition
    /// is unrepresentable in the block form: `since` is the block's own edition.)
    #[test]
    fn declarations_are_internally_consistent() {
        let mut edition_ids = BTreeSet::new();
        let mut encoding_ids = BTreeSet::new();
        for declaration in DEFAULT_DECLARATIONS {
            declaration.edition.id.validate().unwrap();
            assert!(
                edition_ids.insert(declaration.edition.id.to_string()),
                "duplicate edition {}",
                declaration.edition.id
            );

            for encoding in declaration.added {
                let inclusion = EditionInclusion::new(*encoding, declaration.edition.id);
                inclusion.validate().unwrap();
                assert!(
                    encoding_ids.insert(inclusion.encoding_id),
                    "encoding {} is included more than once",
                    inclusion.encoding_id
                );
            }
        }
    }

    #[test]
    fn core_editions_are_drafts() {
        let editions = edition_session();
        assert!(editions.find(&CORE_2026_07_0).is_some_and(|e| e.is_draft()));
        // Drafts are never current: nothing is released yet.
        assert!(editions.current("core").is_none());
    }

    #[test]
    fn default_session_declares_editions() {
        use crate::VortexSessionDefault;
        let session = <VortexSession as VortexSessionDefault>::default();
        assert!(session.editions().find(&CORE_2026_01_0).is_some());
        assert!(session.editions().validate().is_ok());
    }

    #[test]
    fn later_editions_inherit_membership() {
        let editions = edition_session();
        let core_next = EditionId::new("core", 2026, 10, 0);
        let members = editions.encodings_in(&core_next);
        let ids: Vec<&str> = members.iter().map(|i| i.encoding_id.as_str()).collect();
        assert_eq!(ids, FROZEN_CORE_2026_07_0);

        let geo = EditionId::new("geo", 2026, 12, 0);
        assert!(editions.encodings_in(&geo).is_empty());
    }
}
