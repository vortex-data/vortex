// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`Scalar`] wrapper with identity equality.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use crate::scalar::Scalar;

/// A [`Scalar`] that compares by identity, so that two nulls of the same dtype are equal.
///
/// [`Scalar`]'s own [`PartialEq`] follows comparison semantics, under which a null is equal to
/// nothing, so [`Scalar`] cannot implement [`Eq`]. Wrap a scalar in a `ScalarKey` where it must be
/// reflexively equal to itself, such as a hash map key or the options of an expression. Equality
/// uses [`Scalar::eq_nulls_equal`].
///
/// `ScalarKey` dereferences to the wrapped [`Scalar`].
#[derive(Clone, Debug)]
pub struct ScalarKey(Scalar);

impl ScalarKey {
    /// Wraps `scalar`.
    pub fn new(scalar: Scalar) -> Self {
        Self(scalar)
    }

    /// Returns the wrapped scalar.
    pub fn scalar(&self) -> &Scalar {
        &self.0
    }

    /// Unwraps the scalar.
    pub fn into_scalar(self) -> Scalar {
        self.0
    }
}

impl Deref for ScalarKey {
    type Target = Scalar;

    fn deref(&self) -> &Scalar {
        &self.0
    }
}

impl AsRef<Scalar> for ScalarKey {
    fn as_ref(&self) -> &Scalar {
        &self.0
    }
}

impl From<Scalar> for ScalarKey {
    fn from(scalar: Scalar) -> Self {
        Self(scalar)
    }
}

impl From<ScalarKey> for Scalar {
    fn from(key: ScalarKey) -> Self {
        key.0
    }
}

impl PartialEq for ScalarKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq_nulls_equal(&other.0)
    }
}

impl Eq for ScalarKey {}

impl Hash for ScalarKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Display for ScalarKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
