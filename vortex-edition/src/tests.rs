// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ComponentKind;
use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionInclusion;
use crate::EditionMember;
use crate::EditionSession;
use crate::EditionSessionExt;
use crate::EnabledEditions;

const FIRST: EditionId = EditionId::new("test", 2026, 1, 0);
const SECOND: EditionId = EditionId::new("test", 2026, 7, 0);

static DECLARATIONS: &[EditionDeclaration] = &[
    EditionDeclaration {
        edition: Edition {
            id: FIRST,
            min_vortex_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha"),
            EditionMember::array(&"test.beta"),
        ],
    },
    EditionDeclaration {
        edition: Edition {
            id: SECOND,
            min_vortex_version: None,
        },
        added: &[EditionMember::array(&"test.gamma")],
    },
];

fn session() -> EditionSession {
    let editions = EditionSession::empty();
    for declaration in DECLARATIONS {
        editions
            .declare(declaration)
            .unwrap_or_else(|e| panic!("declaring test editions: {e}"));
    }
    editions
}

#[test]
fn editions_pass_the_test_harness() -> Result<(), crate::EditionError> {
    crate::test_harness::validate_edition(&session(), &FIRST)?;
    crate::test_harness::validate_edition(&session(), &SECOND)?;

    // An undeclared edition fails the harness.
    let undeclared = EditionId::new("test", 2027, 1, 0);
    assert!(crate::test_harness::validate_edition(&session(), &undeclared).is_err());
    Ok(())
}

#[test]
fn membership_is_transitive() {
    let editions = session();

    let first = editions.members_in(&FIRST);
    let ids: Vec<&str> = first.iter().map(|i| i.component_id.as_str()).collect();
    assert_eq!(ids, ["test.alpha", "test.beta"]);

    // Members of the first edition are members of the second by inheritance, with their
    // `since` still recording the edition they actually joined in.
    let second = editions.members_in(&SECOND);
    let ids: Vec<&str> = second.iter().map(|i| i.component_id.as_str()).collect();
    assert_eq!(ids, ["test.alpha", "test.beta", "test.gamma"]);
    assert!(
        second
            .iter()
            .filter(|i| i.component_id.as_str() != "test.gamma")
            .all(|i| i.since == FIRST)
    );

    // The second edition's delta is exactly the members declared at it.
    let added: Vec<&str> = second
        .iter()
        .filter(|i| i.since == SECOND)
        .map(|i| i.component_id.as_str())
        .collect();
    assert_eq!(added, ["test.gamma"]);

    // Inheritance never flows backwards, extends to later editions of the family, and
    // never crosses families.
    assert!(first.iter().all(|i| i.since == FIRST));
    let third = EditionId::new("test", 2026, 10, 0);
    assert_eq!(editions.members_in(&third).len(), 3);
    let other = EditionId::new("other", 2026, 10, 0);
    assert!(editions.members_in(&other).is_empty());
}

#[test]
fn drafts_and_current() {
    let editions = session();
    // Both editions are unversioned drafts: neither is current.
    assert!(editions.find(&FIRST).is_some_and(|e| e.is_draft()));
    assert!(editions.current("test").is_none());

    // Freezing the first edition makes it current; the second stays a draft.
    let editions = EditionSession::empty();
    editions
        .declare_edition(Edition {
            id: FIRST,
            min_vortex_version: Some("0.60.0"),
        })
        .unwrap();
    editions
        .declare_edition(Edition {
            id: SECOND,
            min_vortex_version: None,
        })
        .unwrap();
    assert!(editions.validate().is_ok());
    assert_eq!(editions.current("test").map(|e| e.id), Some(FIRST));
}

#[test]
fn session_exposes_edition_registry() {
    // The session variable starts empty; declarations are seeded at initialization time
    // (the `vortex` facade seeds the first-party ones).
    let session = VortexSession::empty().with::<EditionSession>();
    assert!(session.editions().find(&FIRST).is_none());

    for declaration in DECLARATIONS {
        session.editions().declare(declaration).unwrap();
    }
    assert!(session.editions().find(&FIRST).is_some());
}

#[test]
fn registered_and_enabled_editions_are_separate() -> Result<(), crate::EditionError> {
    let session = VortexSession::empty().with::<EditionSession>();
    for declaration in DECLARATIONS {
        session.register_edition(declaration)?;
    }

    assert!(session.enabled_array_encoding_ids().is_empty());
    session.enable_edition(FIRST)?;
    assert_eq!(session.enabled_editions().editions(), [FIRST]);
    assert_eq!(
        session
            .enabled_array_encoding_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        ["test.alpha", "test.beta"]
    );

    session.enable_edition(SECOND)?;
    assert_eq!(session.enabled_editions().editions(), [SECOND]);
    assert_eq!(session.enabled_array_encoding_ids().len(), 3);

    // Selecting an older edition in the same family replaces the newer one and removes
    // encodings that joined after it.
    session.enable_edition(FIRST)?;
    assert_eq!(session.enabled_editions().editions(), [FIRST]);
    let enabled = session.enabled_array_encoding_ids();
    assert_eq!(enabled.len(), 2);
    assert!(enabled.iter().all(|id| id.as_str() != "test.gamma"));
    Ok(())
}

#[test]
fn enabling_requires_registration() {
    let session = VortexSession::empty().with::<EnabledEditions>();
    assert!(session.enable_edition(FIRST).is_err());
    assert!(session.enabled_editions().editions().is_empty());
}

#[test]
fn enabled_editions_are_independent_across_families() -> Result<(), crate::EditionError> {
    const OTHER: EditionId = EditionId::new("other", 2026, 4, 0);
    static OTHER_DECLARATION: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: OTHER,
            min_vortex_version: None,
        },
        added: &[EditionMember::array(&"other.delta")],
    };

    let session = VortexSession::empty().with::<EditionSession>();
    session.register_edition(&DECLARATIONS[0])?;
    session.register_edition(&OTHER_DECLARATION)?;
    session.enable_edition(FIRST)?;
    session.enable_edition(OTHER)?;

    let mut enabled = session.enabled_editions().editions();
    enabled.sort_unstable();
    assert_eq!(enabled, [OTHER, FIRST]);
    assert_eq!(session.enabled_array_encoding_ids().len(), 3);
    Ok(())
}

#[test]
fn duplicate_declarations_error() {
    let editions = session();
    assert!(
        editions
            .declare_edition(Edition {
                id: FIRST,
                min_vortex_version: None,
            })
            .is_err()
    );
    assert!(
        editions
            .declare_inclusion(EditionInclusion::array("test.alpha", FIRST))
            .is_err()
    );
}

#[test]
fn validate_rejects_inconsistent_declarations() -> Result<(), crate::EditionError> {
    // An inclusion referencing an undeclared edition.
    let editions = EditionSession::empty();
    editions.declare_inclusion(EditionInclusion::array("test.alpha", FIRST))?;
    assert!(editions.validate().is_err());

    // A member requiring a release newer than its edition declares.
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: FIRST,
        min_vortex_version: Some("0.70.0"),
    })?;
    editions.declare_inclusion(EditionInclusion {
        required_vortex_release: Some("0.80.0"),
        ..EditionInclusion::array("test.alpha", FIRST)
    })?;
    assert!(editions.validate().is_err());

    // A frozen edition following a draft within the same family.
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: FIRST,
        min_vortex_version: None,
    })?;
    editions.declare_edition(Edition {
        id: SECOND,
        min_vortex_version: Some("0.70.0"),
    })?;
    assert!(editions.validate().is_err());

    // A malformed edition id (family not lowercase, month out of range).
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: EditionId::new("Test", 2026, 13, 0),
        min_vortex_version: None,
    })?;
    assert!(editions.validate().is_err());

    // A malformed encoding id.
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: FIRST,
        min_vortex_version: None,
    })?;
    editions.declare_inclusion(EditionInclusion::array("Test.ALPHA", FIRST))?;
    assert!(editions.validate().is_err());

    Ok(())
}

#[test]
fn edition_ids_order_within_family_only() {
    assert!(FIRST.is_at_or_before(&SECOND));
    assert!(!SECOND.is_at_or_before(&FIRST));

    let other = EditionId::new("other", 2026, 3, 0);
    assert!(!FIRST.is_at_or_before(&other));
    assert!(!other.is_at_or_before(&FIRST));
}

#[test]
fn edition_id_display() {
    assert_eq!(FIRST.to_string(), "test2026.01.0");
}

#[test]
fn members_carry_their_component_kind() -> Result<(), crate::EditionError> {
    static MIXED: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: FIRST,
            min_vortex_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha"),
            EditionMember::layout(&"test.alpha"),
            EditionMember::aggregate_fn(&"test.sum"),
            EditionMember::scalar_fn(&"test.add"),
        ],
    };

    let editions = EditionSession::empty();
    editions.declare(&MIXED)?;
    editions.validate()?;

    // The same id under two kinds is two distinct members, and each keeps its kind.
    let members = editions.members_in(&FIRST);
    assert_eq!(members.len(), 4);
    let alphas: Vec<ComponentKind> = members
        .iter()
        .filter(|i| i.component_id.as_str() == "test.alpha")
        .map(|i| i.kind)
        .collect();
    assert_eq!(alphas, [ComponentKind::Array, ComponentKind::Layout]);

    // Kinds are resolved one at a time, so a layout never reaches the array registry.
    let arrays = editions.components_in(&FIRST, ComponentKind::Array);
    assert_eq!(arrays.len(), 1);
    assert_eq!(arrays[0].component_id.as_str(), "test.alpha");

    // A duplicate within one kind is still an error.
    assert!(
        editions
            .declare_inclusion(EditionInclusion::array("test.alpha", FIRST))
            .is_err()
    );
    Ok(())
}

#[test]
fn enabled_ids_are_resolved_per_kind() -> Result<(), crate::EditionError> {
    static MIXED: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: FIRST,
            min_vortex_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha"),
            EditionMember::layout(&"test.flat"),
            EditionMember::aggregate_fn(&"test.sum"),
        ],
    };

    let session = VortexSession::empty().with::<EditionSession>();
    session.register_edition(&MIXED)?;
    session.enable_edition(FIRST)?;

    let ids = |kind| {
        session
            .enabled_component_ids(kind)
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(ComponentKind::Array), ["test.alpha"]);
    assert_eq!(ids(ComponentKind::Layout), ["test.flat"]);
    assert_eq!(ids(ComponentKind::AggregateFn), ["test.sum"]);
    assert!(ids(ComponentKind::ScalarFn).is_empty());

    // What a writer may emit is the array set alone.
    assert_eq!(
        session
            .enabled_array_encoding_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        ["test.alpha"]
    );
    Ok(())
}
