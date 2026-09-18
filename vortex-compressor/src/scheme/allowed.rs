// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The serialized IDs a writer permits a compressor to emit.

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_set::HashSet;

/// The serialized IDs a compressor may write under.
///
/// The file writer derives this from its enabled editions. The compressor drops schemes whose
/// [`produced_encodings`](crate::scheme::Scheme::produced_encodings) it does not permit and hands
/// it to the rest through [`CompressorContext`](crate::scheme::CompressorContext).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AllowedSerializedIds {
    /// Unrestricted. Every scheme resolves to its newest version.
    #[default]
    All,
    /// Only these IDs.
    Only(HashSet<ArrayId>),
}

impl AllowedSerializedIds {
    /// Whether the writer may emit `id`.
    pub fn permits(&self, id: &ArrayId) -> bool {
        match self {
            Self::All => true,
            Self::Only(ids) => ids.contains(id),
        }
    }

    /// Whether the writer may emit every one of `ids`.
    pub fn permits_all(&self, ids: &[ArrayId]) -> bool {
        ids.iter().all(|id| self.permits(id))
    }

    /// Narrows the permitted IDs to those also permitted by `other`.
    pub fn intersect(&mut self, other: &Self) {
        if let Self::Only(ids) = other {
            self.restrict(ids);
        }
    }

    /// Narrows the permitted IDs to those also in `allowed`.
    pub fn restrict(&mut self, allowed: &HashSet<ArrayId>) {
        *self = match self {
            Self::All => Self::Only(allowed.clone()),
            Self::Only(existing) => Self::Only(existing.intersection(allowed).copied().collect()),
        };
    }
}
