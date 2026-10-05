// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test harness for edition declarations.
//!
//! Each edition definition should call [`validate_edition`] once from its `#[cfg(test)]`
//! module, so every declared edition has a test proving its constraints hold.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::EditionId;
use crate::EditionSession;

/// Validate one edition's constraints: its identifier is well-formed, it is declared in the
/// given session, and the session's declarations as a whole validate (chronology, version
/// forms, membership constraints).
///
/// ```ignore
/// #[cfg(test)]
/// mod tests {
///     #[test]
///     fn edition_is_valid() -> vortex_error::VortexResult<()> {
///         vortex_edition::test_harness::validate_edition(&edition_session(), &CORE_2026_01_0)
///     }
/// }
/// ```
pub fn validate_edition(editions: &EditionSession, edition: &EditionId) -> VortexResult<()> {
    edition.validate()?;
    if editions.find(edition).is_none() {
        vortex_bail!("{edition} is not declared in the session");
    }
    editions.validate()
}
