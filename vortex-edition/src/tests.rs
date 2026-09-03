// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ComponentKind;
use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionFamily;
use crate::EditionId;
use crate::EditionInclusion;
use crate::EditionMember;
use crate::EditionSession;
use crate::EditionSessionExt;
use crate::EnabledEditions;

static TEST_FAMILY: EditionFamily = EditionFamily {
    name: "test",
    origin: "vortex-edition-tests",
    doc: "A family used by the unit tests.",
};

static OTHER_FAMILY: EditionFamily = EditionFamily {
    name: "other",
    origin: "vortex-edition-tests",
    doc: "A second family, for checking that families stay independent.",
};

const FIRST: EditionId = EditionId::new("test", 2026, 1, 0);
const SECOND: EditionId = EditionId::new("test", 2026, 7, 0);

static DECLARATIONS: &[EditionDeclaration] = &[
    EditionDeclaration {
        edition: Edition {
            id: FIRST,
            min_library_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha"),
            EditionMember::array(&"test.beta"),
        ],
    },
    EditionDeclaration {
        edition: Edition {
            id: SECOND,
            min_library_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha_v2"),
            EditionMember::array(&"test.gamma"),
        ],
    },
];

fn session() -> EditionSession {
    let editions = EditionSession::empty();
    editions
        .declare_family(&TEST_FAMILY)
        .unwrap_or_else(|e| panic!("declaring the test family: {e}"));
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
fn membership_is_transitive() -> Result<(), crate::EditionError> {
    let editions = session();

    let first = editions.components_in(&FIRST, ComponentKind::Array);
    let ids: Vec<&str> = first.iter().map(|i| i.component_id.as_str()).collect();
    assert_eq!(ids, ["test.alpha", "test.beta"]);

    // Members of the first edition are inherited. A newer wire representation has its own ID, so
    // both the historical and current representations remain explicit members.
    let second = editions.components_in(&SECOND, ComponentKind::Array);
    let ids: Vec<&str> = second.iter().map(|i| i.component_id.as_str()).collect();
    assert_eq!(
        ids,
        ["test.alpha", "test.alpha_v2", "test.beta", "test.gamma"]
    );
    let alpha = second
        .iter()
        .find(|i| i.component_id.as_str() == "test.alpha")
        .ok_or_else(|| crate::EditionError::new("test.alpha is a member"))?;
    assert_eq!(alpha.since, FIRST);
    let alpha_v2 = second
        .iter()
        .find(|i| i.component_id.as_str() == "test.alpha_v2")
        .ok_or_else(|| crate::EditionError::new("test.alpha_v2 is a member"))?;
    assert_eq!(alpha_v2.since, SECOND);
    let beta = second
        .iter()
        .find(|i| i.component_id.as_str() == "test.beta")
        .ok_or_else(|| crate::EditionError::new("test.beta is a member"))?;
    assert_eq!(beta.since, FIRST);

    // The second edition's delta is exactly the members declared at it.
    let added: Vec<&str> = second
        .iter()
        .filter(|i| i.since == SECOND)
        .map(|i| i.component_id.as_str())
        .collect();
    assert_eq!(added, ["test.alpha_v2", "test.gamma"]);

    // Inheritance never flows backwards, extends to later editions of the family, and
    // never crosses families.
    assert!(first.iter().all(|i| i.since == FIRST));
    let third = EditionId::new("test", 2026, 10, 0);
    assert_eq!(
        editions.components_in(&third, ComponentKind::Array).len(),
        4
    );
    let other = EditionId::new("other", 2026, 10, 0);
    assert!(
        editions
            .components_in(&other, ComponentKind::Array)
            .is_empty()
    );
    Ok(())
}

#[test]
fn drafts_and_current() {
    let editions = session();
    // Both editions are unversioned drafts: neither is current.
    assert!(editions.find(&FIRST).is_some_and(|e| e.is_draft()));
    assert!(editions.current("test").is_none());

    // Freezing the first edition makes it current; the second stays a draft.
    let editions = EditionSession::empty();
    editions.declare_family(&TEST_FAMILY).unwrap();
    editions
        .declare_edition(Edition {
            id: FIRST,
            min_library_version: Some("0.60.0"),
        })
        .unwrap();
    editions
        .declare_edition(Edition {
            id: SECOND,
            min_library_version: None,
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

    assert!(
        session
            .enabled_component_ids(ComponentKind::Array)
            .is_empty()
    );
    session.enable_edition(FIRST)?;
    assert_eq!(session.enabled_editions().editions(), [FIRST]);
    assert_eq!(
        session
            .enabled_component_ids(ComponentKind::Array)
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        ["test.alpha", "test.beta"]
    );
    session.enable_edition(SECOND)?;
    assert_eq!(session.enabled_editions().editions(), [SECOND]);
    assert_eq!(session.enabled_component_ids(ComponentKind::Array).len(), 4);

    // Selecting an older edition in the same family replaces the newer one and removes
    // encodings that joined after it.
    session.enable_edition(FIRST)?;
    assert_eq!(session.enabled_editions().editions(), [FIRST]);
    let enabled = session.enabled_component_ids(ComponentKind::Array);
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
            min_library_version: None,
        },
        added: &[EditionMember::array(&"other.delta")],
    };

    let session = VortexSession::empty().with::<EditionSession>();
    session.editions().declare_family(&TEST_FAMILY)?;
    session.editions().declare_family(&OTHER_FAMILY)?;
    session.register_edition(&DECLARATIONS[0])?;
    session.register_edition(&OTHER_DECLARATION)?;
    session.editions().validate()?;
    session.enable_edition(FIRST)?;
    session.enable_edition(OTHER)?;

    let mut enabled = session.enabled_editions().editions();
    enabled.sort_unstable();
    assert_eq!(enabled, [OTHER, FIRST]);
    assert_eq!(session.enabled_component_ids(ComponentKind::Array).len(), 3);
    Ok(())
}

#[test]
fn serialized_array_ids_can_be_added_by_an_opt_in_family() -> Result<(), crate::EditionError> {
    const OPT_IN: EditionId = EditionId::new("other", 2026, 8, 0);
    static OPT_IN_DECLARATION: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: OPT_IN,
            min_library_version: None,
        },
        added: &[EditionMember::array(&"test.alpha_v2")],
    };

    let session = VortexSession::empty().with::<EditionSession>();
    session.editions().declare_family(&TEST_FAMILY)?;
    session.editions().declare_family(&OTHER_FAMILY)?;
    session.register_edition(&DECLARATIONS[0])?;
    session.register_edition(&OPT_IN_DECLARATION)?;
    session.editions().validate()?;
    session.enable_edition(FIRST)?;
    session.enable_edition(OPT_IN)?;

    assert_eq!(
        session
            .enabled_component_ids(ComponentKind::Array)
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        ["test.alpha", "test.alpha_v2", "test.beta"]
    );
    Ok(())
}

#[test]
fn duplicate_declarations_error() {
    let editions = session();
    assert!(
        editions
            .declare_edition(Edition {
                id: FIRST,
                min_library_version: None,
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
        min_library_version: Some("0.70.0"),
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
        min_library_version: None,
    })?;
    editions.declare_edition(Edition {
        id: SECOND,
        min_library_version: Some("0.70.0"),
    })?;
    assert!(editions.validate().is_err());

    // A malformed edition id (family not lowercase, month out of range).
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: EditionId::new("Test", 2026, 13, 0),
        min_library_version: None,
    })?;
    assert!(editions.validate().is_err());

    // A malformed encoding id.
    let editions = EditionSession::empty();
    editions.declare_edition(Edition {
        id: FIRST,
        min_library_version: None,
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
fn families_must_be_declared_before_their_editions() -> Result<(), crate::EditionError> {
    // An edition whose family was never declared: the name would otherwise be whatever the
    // declaration happened to spell, and a typo would mint a family of one.
    let editions = EditionSession::empty();
    editions.declare(&DECLARATIONS[0])?;
    assert!(editions.validate().is_err());

    editions.declare_family(&TEST_FAMILY)?;
    editions.validate()?;

    // Declaring the same family twice is an error, as it is for editions.
    assert!(editions.declare_family(&TEST_FAMILY).is_err());
    Ok(())
}

#[test]
fn families_must_document_themselves() {
    let editions = EditionSession::empty();
    editions
        .declare_family(&EditionFamily {
            name: "undocumented",
            origin: "vortex-edition-tests",
            doc: "  ",
        })
        .unwrap();
    assert!(editions.validate().is_err());
}

#[test]
fn families_must_name_their_origin() {
    let editions = EditionSession::empty();
    editions
        .declare_family(&EditionFamily {
            name: "unowned",
            origin: "  ",
            doc: "A family without an origin.",
        })
        .unwrap();
    let error = editions.validate().unwrap_err();
    assert!(error.to_string().contains("origin library or project"));
}

#[test]
fn kinds_are_resolved_independently() -> Result<(), crate::EditionError> {
    // `test.alpha` is declared under both kinds: same id, two distinct members.
    static MIXED: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: FIRST,
            min_library_version: None,
        },
        added: &[
            EditionMember::array(&"test.alpha"),
            EditionMember::dtype(&"test.alpha"),
            EditionMember::layout(&"test.alpha"),
            EditionMember::layout(&"test.flat"),
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
    // A layout never reaches the array registry, and what a writer may emit is the arrays.
    assert_eq!(ids(ComponentKind::Array), ["test.alpha"]);
    assert_eq!(ids(ComponentKind::DType), ["test.alpha"]);
    assert_eq!(ids(ComponentKind::Layout), ["test.alpha", "test.flat"]);
    assert_eq!(session.enabled_component_ids(ComponentKind::Array).len(), 1);

    // A duplicate within one kind is still an error.
    assert!(
        session
            .editions()
            .declare_inclusion(EditionInclusion::array("test.alpha", FIRST))
            .is_err()
    );
    Ok(())
}
