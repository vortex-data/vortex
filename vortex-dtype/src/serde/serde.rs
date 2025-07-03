// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "serde")]

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Nullability;

/// Serialize Nullability as a boolean
impl Serialize for Nullability {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        bool::from(*self).serialize(serializer)
    }
}

/// Deserialize Nullability from a boolean
impl<'de> Deserialize<'de> for Nullability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        bool::deserialize(deserializer).map(Self::from)
    }
}
