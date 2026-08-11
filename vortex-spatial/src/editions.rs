// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `spatial` edition family.
//!
//! Spatial support is opt-in: a reader without this crate cannot resolve `vortex.st.*`, so
//! spatial members live in their own family rather than in `core`. [`crate::initialize`]
//! registers and enables the edition, because a session that registers the AABB zone stat and
//! is then forbidden from writing it would compute pruning statistics it must throw away.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The August 2026 draft edition of the `spatial` family.
pub const SPATIAL_2026_08: EditionId = EditionId::new("spatial", 2026, 8, 0);

/// The declaration of [`SPATIAL_2026_08`] and the components that join the family at it.
///
/// A draft: no Vortex release yet guarantees these members forever. The AABB aggregate is
/// recorded in the zone maps of geometry columns, so it must be a member for a session with
/// spatial support to write those columns.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: SPATIAL_2026_08,
        min_vortex_version: None,
    },
    added: &[EditionMember::aggregate(&"vortex.st.aabb")],
};

#[cfg(test)]
mod tests {
    use vortex_edition::ComponentKind;
    use vortex_edition::EditionError;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::test_harness::validate_edition;

    use super::*;

    #[test]
    fn spatial_edition_is_valid() -> Result<(), EditionError> {
        let session = crate::test_harness::spatial_session();
        validate_edition(&session.editions(), &SPATIAL_2026_08)
    }

    /// A spatial session must permit the AABB zone stat it registers, or writing a geometry
    /// column fails on the aggregate the session itself asked for.
    #[test]
    fn initialize_permits_the_aabb_zone_stat() {
        let session = crate::test_harness::spatial_session();
        let enabled = session.enabled_component_ids(ComponentKind::Aggregate);
        assert!(
            enabled.iter().any(|id| id.as_str() == "vortex.st.aabb"),
            "spatial session permits {enabled:?}"
        );
    }
}
