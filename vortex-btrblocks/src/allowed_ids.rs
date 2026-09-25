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
/// An ID is allowed when it passes every restriction: it is in `registered` and `editions` (when
/// set) and not in `excluded`. A scheme is kept only if every ID it declares in
/// [`produced_encodings`](crate::Scheme::produced_encodings) is allowed.
///
/// The restrictions are independent, so lifting one never undoes another.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AllowedIds {
    /// The IDs of the session's registered array plugins, or `None` for no restriction.
    pub(crate) registered: Option<HashSet<ArrayId>>,
    /// The IDs the session's enabled editions permit, or `None` for no restriction.
    pub(crate) editions: Option<HashSet<ArrayId>>,
    /// IDs excluded by a builder preset, such as the CUDA one. Always applied.
    pub(crate) excluded: HashSet<ArrayId>,
}

impl AllowedIds {
    /// Allows every serialized ID.
    pub(crate) fn all() -> Self {
        Self::default()
    }

    /// Allows the serialized array IDs that are registered in the session and permitted by its
    /// enabled editions, matching what the file writer can write.
    ///
    /// A session with no enabled editions permits no IDs.
    pub(crate) fn from_session(session: &VortexSession) -> Self {
        let registered = session
            .arrays()
            .registry()
            .read(|registry| registry.keys().copied().collect());
        let editions = session
            .enabled_component_ids(ComponentKind::Array)
            .into_iter()
            .collect();
        Self {
            registered: Some(registered),
            editions: Some(editions),
            excluded: HashSet::default(),
        }
    }

    /// Returns whether `id` is allowed.
    pub(crate) fn is_allowed(&self, id: &ArrayId) -> bool {
        !self.excluded.contains(id)
            && self.registered.as_ref().is_none_or(|ids| ids.contains(id))
            && self.editions.as_ref().is_none_or(|ids| ids.contains(id))
    }
}
