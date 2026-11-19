// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Membership of the protobuf dtype in the `unstable` edition, and its use of the extension layout.

use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_edition::UNSTABLE;
use vortex_edition::test_harness::validate_edition;
use vortex_error::VortexResult;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

use crate::Protobuf;

/// Returns the ids of the extension dtypes that `session` may write, in sorted order.
fn writable_dtypes(session: &VortexSession) -> Vec<String> {
    let mut ids = session
        .enabled_component_ids(ComponentKind::DType)
        .iter()
        .map(|id| id.as_str().to_owned())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

#[test]
fn test_initialize_declares_unstable_membership() -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    // Initializing twice must not declare the membership twice.
    crate::initialize(&session);
    validate_edition(&session.editions(), &UNSTABLE)?;

    // JSON, which protobuf columns hold under the JSON text mapping, has its own edition, which
    // initializing JSON support enables.
    assert_eq!(writable_dtypes(&session), ["vortex.json"]);
    session.enable_edition(UNSTABLE)?;
    assert_eq!(writable_dtypes(&session), ["vortex.json", Protobuf::ID]);
    Ok(())
}

#[test]
fn test_initialize_opts_into_extension_layout() {
    let session = VortexSession::empty();
    assert!(session.layouts().decomposed_extensions().is_empty());
    crate::initialize(&session);
    let ids = session.layouts().decomposed_extensions();
    assert_eq!(
        ids.iter().map(|id| id.as_str()).collect::<Vec<_>>(),
        [Protobuf::ID]
    );
}
