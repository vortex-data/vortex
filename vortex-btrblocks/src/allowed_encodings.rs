// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialized ID permissions for compression schemes.

use vortex_array::ArrayId;
use vortex_utils::aliases::hash_set::HashSet;

/// The serialized IDs a compressor may write.
///
/// Permissions usually start as [`Only`](Self::Only) the array IDs a session's enabled editions
/// permit. [`allow_all`](Self::allow_all) lifts the allowlist for in-memory compression, where no
/// edition restricts what a file may contain. [`disallow`](Self::disallow) denies IDs under either
/// variant, and a denial survives a later `allow_all`. The builder drops every scheme declaring
/// an ID that is not permitted.
#[derive(Clone, Debug)]
pub enum AllowedEncodings {
    /// Every serialized ID except `denied`.
    All {
        /// IDs that are never permitted.
        denied: HashSet<ArrayId>,
    },
    /// Exactly `allowed`, minus `denied`.
    Only {
        /// Permitted IDs.
        allowed: HashSet<ArrayId>,
        /// IDs that are never permitted.
        denied: HashSet<ArrayId>,
    },
}

impl Default for AllowedEncodings {
    /// Permits nothing.
    fn default() -> Self {
        Self::only([])
    }
}

impl AllowedEncodings {
    /// Permissions for exactly `allowed`.
    pub fn only(allowed: impl IntoIterator<Item = ArrayId>) -> Self {
        Self::Only {
            allowed: allowed.into_iter().collect(),
            denied: HashSet::new(),
        }
    }

    /// Permissions for every serialized ID.
    pub fn all() -> Self {
        Self::All {
            denied: HashSet::new(),
        }
    }

    /// Permits `ids`, lifting any earlier denial.
    pub fn allow(&mut self, ids: impl IntoIterator<Item = ArrayId>) {
        match self {
            Self::All { denied } => {
                for id in ids {
                    denied.remove(&id);
                }
            }
            Self::Only { allowed, denied } => {
                for id in ids {
                    denied.remove(&id);
                    allowed.insert(id);
                }
            }
        }
    }

    /// Permits every serialized ID that is not denied.
    pub fn allow_all(&mut self) {
        let denied = match self {
            Self::All { denied } | Self::Only { denied, .. } => std::mem::take(denied),
        };
        *self = Self::All { denied };
    }

    /// Denies `ids`, even after [`allow_all`](Self::allow_all).
    pub fn disallow(&mut self, ids: impl IntoIterator<Item = ArrayId>) {
        match self {
            Self::All { denied } => denied.extend(ids),
            Self::Only { allowed, denied } => {
                for id in ids {
                    allowed.remove(&id);
                    denied.insert(id);
                }
            }
        }
    }

    /// Whether arrays serialized under `id` may be written.
    pub fn permits(&self, id: &ArrayId) -> bool {
        match self {
            Self::All { denied } => !denied.contains(id),
            Self::Only { allowed, denied } => allowed.contains(id) && !denied.contains(id),
        }
    }

    /// Whether every ID in `ids` may be written.
    pub fn permits_all(&self, ids: &[ArrayId]) -> bool {
        ids.iter().all(|id| self.permits(id))
    }
}

#[cfg(test)]
mod tests {
    use vortex_session::registry::CachedId;

    use super::*;

    static A: CachedId = CachedId::new("vortex.test.a");
    static B: CachedId = CachedId::new("vortex.test.b");
    static C: CachedId = CachedId::new("vortex.test.c");

    #[test]
    fn default_permits_nothing() {
        assert!(!AllowedEncodings::default().permits(&A));
    }

    #[test]
    fn only_permits_the_listed_ids() {
        let mut allowed = AllowedEncodings::only([*A]);
        assert!(allowed.permits(&A));
        assert!(!allowed.permits(&B));
        allowed.allow([*B]);
        assert!(allowed.permits_all(&[*A, *B]));
    }

    #[test]
    fn denial_survives_allow_all() {
        let mut allowed = AllowedEncodings::only([*A, *B]);
        allowed.disallow([*B]);
        allowed.allow_all();
        assert!(allowed.permits(&A));
        assert!(allowed.permits(&C));
        assert!(!allowed.permits(&B));
    }

    #[test]
    fn allow_lifts_a_denial() {
        let mut allowed = AllowedEncodings::all();
        allowed.disallow([*A]);
        assert!(!allowed.permits(&A));
        allowed.allow([*A]);
        assert!(allowed.permits(&A));
    }
}
