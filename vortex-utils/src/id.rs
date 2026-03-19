// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A lightweight interned identifier type backed by `&'static str`.

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

/// A lightweight, copyable identifier backed by a `&'static str`.
///
/// Used for array encoding IDs, scalar function IDs, layout IDs, and similar
/// globally-unique string identifiers throughout Vortex.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(&'static str);

impl Id {
    /// Create a new `Id` from a static string.
    pub const fn new(s: &'static str) -> Self {
        Self(s)
    }

    /// Returns the underlying string.
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Debug for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Id(\"{}\")", self.0)
    }
}

impl PartialOrd for Id {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Id {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(other.0)
    }
}

impl PartialEq<str> for Id {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl AsRef<str> for Id {
    fn as_ref(&self) -> &str {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_equality() {
        let a = Id::new("vortex.primitive");
        let b = Id::new("vortex.primitive");
        let c = Id::new("vortex.bool");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_id_display() {
        let id = Id::new("vortex.primitive");
        assert_eq!(format!("{id}"), "vortex.primitive");
    }

    #[test]
    fn test_id_debug() {
        let id = Id::new("vortex.primitive");
        assert_eq!(format!("{id:?}"), "Id(\"vortex.primitive\")");
    }

    #[test]
    fn test_id_partial_eq_str() {
        let id = Id::new("vortex.primitive");
        assert_eq!(id, *"vortex.primitive");
    }

    #[test]
    fn test_id_ord() {
        let a = Id::new("aaa");
        let b = Id::new("bbb");
        assert!(a < b);
    }

    #[test]
    fn test_id_const() {
        const MY_ID: Id = Id::new("test");
        assert_eq!(MY_ID.as_str(), "test");
    }
}
