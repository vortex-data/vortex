// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use serde::Deserialize;
use serde::Serialize;

/// A benchmark entry, grouped by benchmark group, then chart name, then series name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    // `StructArray`
    pub commit_id: CommitId,     // fixed size list of `u8` (20 bytes)
    pub benchmark_group: NameId, // `u32` array
    pub chart_name: NameId,      // `u32` array
    pub series_name: NameId,     // `u32` array
    pub value: u64,              // `u64` array
}

/// String ID lookup so that we don't have to store the string every time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NameId(pub u32);

/// The 20-byte binary SHA-1 Git commit ID.
#[derive(Clone, PartialEq, Eq)]
pub struct CommitId(pub [u8; 20]);

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
