// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Git commit ID type with passthrough hashing.

use std::fmt;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;

use serde::Deserialize;
use serde::Serialize;

/// The 20-byte binary SHA-1 Git commit ID.
///
/// Note that the ordering of commit IDs does not really mean anything, we just have it implemented
/// for convenience.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommitId(pub [u8; 20]);

impl Serialize for CommitId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for CommitId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CommitIdVisitor;

        impl<'de> serde::de::Visitor<'de> for CommitIdVisitor {
            type Value = CommitId;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 40-character hexadecimal string")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value.len() != 40 {
                    return Err(E::custom(format!(
                        "expected 40 hex characters, got {}",
                        value.len()
                    )));
                }

                let bytes = hex::decode(value)
                    .map_err(|e| E::custom(format!("invalid hexadecimal: {}", e)))?;

                let mut arr = [0u8; 20];
                arr.copy_from_slice(&bytes);
                Ok(CommitId(arr))
            }
        }

        deserializer.deserialize_str(CommitIdVisitor)
    }
}

impl Hash for CommitId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0);
    }
}

impl fmt::Display for CommitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl fmt::Debug for CommitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommitId(\"{}\")", hex::encode(self.0))
    }
}

/// A hasher that passes through bytes directly without additional hashing.
///
/// This is useful for types like [`CommitId`] that are already cryptographic hashes.
#[derive(Default)]
pub struct PassthroughHasher(u64);

impl Hasher for PassthroughHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Use the first 8 bytes (or fewer) as the hash value.
        let len = bytes.len().min(8);
        let mut buf = [0u8; 8];
        buf[..len].copy_from_slice(&bytes[..len]);
        self.0 = u64::from_le_bytes(buf);
    }
}

/// A [`BuildHasher`] that creates [`PassthroughHasher`] instances.
#[derive(Default, Clone)]
pub struct PassthroughBuildHasher;

impl BuildHasher for PassthroughBuildHasher {
    type Hasher = PassthroughHasher;

    fn build_hasher(&self) -> Self::Hasher {
        PassthroughHasher::default()
    }
}
