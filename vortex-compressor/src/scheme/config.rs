// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Configuration shared by schemes when constructing a compressor.

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_set::HashSet;

/// Configuration used to select each compression scheme's supported behavior.
///
/// By default all serialized IDs are permitted. File writers restrict these to IDs allowed by
/// their enabled editions and registered plugins.
#[derive(Debug, Clone, Default)]
pub struct SchemeConfig {
    /// `None` permits all serialized IDs; an empty set permits none.
    allowed_serialized_ids: Option<HashSet<ArrayId>>,
}

impl SchemeConfig {
    /// Restricts the serialized IDs schemes may emit, intersecting with any earlier restriction.
    pub fn with_allowed_serialized_ids(mut self, allowed: &HashSet<ArrayId>) -> Self {
        match &mut self.allowed_serialized_ids {
            None => self.allowed_serialized_ids = Some(allowed.clone()),
            Some(ids) => ids.retain(|id| allowed.contains(id)),
        }
        self
    }

    /// Whether a scheme may construct an array serialized under `id`.
    pub fn allows_serialized_id(&self, id: &ArrayId) -> bool {
        self.allowed_serialized_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(id))
    }
}
