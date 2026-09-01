// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `json` edition family.
//!
//! JSON support is opt-in. [`crate::initialize`] registers and enables the edition together with
//! the dtype plugin.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The `json` family: the JSON extension dtype.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "json",
    origin: "vortex-json",
    doc: "The JSON extension dtype. JSON support is opt-in: a reader built without \
`vortex-json` cannot resolve `vortex.json`, so the dtype is versioned independently of \
`core` and a session enables this family only by initializing the crate.",
};

/// The August 2026 draft edition of the `json` family.
pub const JSON_2026_08: EditionId = EditionId::new("json", 2026, 8, 0);

/// The declaration of [`JSON_2026_08`] and the components that join the family at it.
///
/// A draft: no Vortex release yet guarantees this member forever. The JSON extension dtype is
/// embedded in file schemas, so it must be enabled for a session with JSON support to write those
/// schemas under an edition policy.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: JSON_2026_08,
        min_library_version: None,
    },
    added: &[EditionMember::dtype(&"vortex.json")],
};

#[cfg(test)]
mod tests {
    use vortex_edition::ComponentKind;
    use vortex_edition::EditionError;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::test_harness::validate_edition;

    use super::*;

    fn json_session() -> vortex_session::VortexSession {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    }

    #[test]
    fn json_edition_is_valid() -> Result<(), EditionError> {
        let session = json_session();
        validate_edition(&session.editions(), &JSON_2026_08)
    }

    #[test]
    fn initialize_permits_the_json_dtype() {
        let session = json_session();
        let enabled = session.enabled_component_ids(ComponentKind::DType);
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].as_str(), "vortex.json");
    }
}
