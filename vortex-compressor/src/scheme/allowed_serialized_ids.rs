// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialized ID permissions used when configuring compression schemes.

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_set::HashSet;

/// The serialized IDs permitted for a configured compression scheme.
///
/// Defaults to permitting every ID. Restrictions and exclusions only narrow the permitted set.
#[derive(Debug, Clone, Default)]
pub enum AllowedSerializedIds {
    /// All IDs are permitted.
    #[default]
    All,
    /// Only these IDs are permitted.
    Only(HashSet<ArrayId>),
    /// All IDs except these are permitted.
    AllExcept(HashSet<ArrayId>),
}

impl AllowedSerializedIds {
    /// Returns whether a serialized ID is permitted.
    pub fn contains(&self, id: &ArrayId) -> bool {
        match self {
            Self::All => true,
            Self::Only(allowed) => allowed.contains(id),
            Self::AllExcept(forbidden) => !forbidden.contains(id),
        }
    }

    /// Restricts the permitted IDs to their intersection with `allowed`.
    ///
    /// Previously excluded IDs remain excluded, and an empty set permits no IDs.
    pub fn restrict_to(&mut self, allowed: &HashSet<ArrayId>) {
        match self {
            Self::All => *self = Self::Only(allowed.clone()),
            Self::Only(current) => current.retain(|id| allowed.contains(id)),
            Self::AllExcept(forbidden) => {
                let permitted = allowed.difference(forbidden).copied().collect();
                *self = Self::Only(permitted);
            }
        }
    }

    /// Excludes an ID, including from subsequent calls to [`Self::restrict_to`].
    pub fn exclude(&mut self, id: ArrayId) {
        match self {
            Self::All => *self = Self::AllExcept(HashSet::from([id])),
            Self::Only(allowed) => {
                allowed.remove(&id);
            }
            Self::AllExcept(forbidden) => {
                forbidden.insert(id);
            }
        }
    }
}
