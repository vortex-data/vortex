// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The serialized array IDs a compressor may produce.

use vortex_array::ArrayId;
use vortex_array::session::ArraySessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

/// The serialized array IDs a compressor's schemes may produce.
///
/// An ID is allowed when it is not in `blacklist` and either `allow_all` is set or the ID is in
/// `allowlist`. A scheme is kept only if every ID it declares in
/// [`produced_encodings`](crate::Scheme::produced_encodings) is allowed.
///
/// Built from a session, the allowlist is the IDs of the registered array plugins and the
/// blacklist is those the session's enabled editions do not permit, matching what the file writer
/// can write.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PermissionSet {
    /// Allow every ID not in `blacklist`, ignoring `allowlist`.
    pub(crate) allow_all: bool,
    /// IDs allowed when `allow_all` is unset.
    pub(crate) allowlist: HashSet<ArrayId>,
    /// IDs never allowed.
    pub(crate) blacklist: HashSet<ArrayId>,
}

impl PermissionSet {
    /// Allows every serialized ID.
    pub(crate) fn all() -> Self {
        Self {
            allow_all: true,
            ..Self::default()
        }
    }

    /// Allows the serialized array IDs that are registered in the session and permitted by its
    /// enabled editions.
    ///
    /// A session with no enabled editions permits no IDs.
    pub(crate) fn from_session(session: &VortexSession) -> Self {
        let allowlist: HashSet<ArrayId> = session
            .arrays()
            .registry()
            .read(|registry| registry.keys().copied().collect());
        let enabled: HashSet<ArrayId> = session
            .enabled_component_ids(ComponentKind::Array)
            .into_iter()
            .collect();
        let blacklist = allowlist.difference(&enabled).copied().collect();
        Self {
            allow_all: false,
            allowlist,
            blacklist,
        }
    }

    /// Returns whether `id` is allowed.
    pub(crate) fn is_allowed(&self, id: &ArrayId) -> bool {
        !self.blacklist.contains(id) && (self.allow_all || self.allowlist.contains(id))
    }
}
