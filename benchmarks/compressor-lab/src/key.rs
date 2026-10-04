// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stable content hashes for task keys, plan ids and file fingerprints.

use std::fmt;
use std::fmt::Write as _;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;

/// The number of hex characters kept in a [`TaskKey`] (128 bits).
const TASK_KEY_HEX_LEN: usize = 32;

/// A task's identity: a hash of its kind, inputs, parameters and code identity.
///
/// Two tasks with the same key do the same work and produce the same facts, whichever plan
/// they appear in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskKey(String);

impl TaskKey {
    /// Derives a key from a task kind and its key material.
    pub fn derive(kind: &str, material: &impl Serialize) -> anyhow::Result<Self> {
        let mut digest = stable_digest(kind, material)?;
        digest.truncate(TASK_KEY_HEX_LEN);
        Ok(Self(digest))
    }

    /// The key as a hex string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A uniformly distributed number derived from the key, used to assign shards.
    pub fn bucket(&self) -> u64 {
        self.0
            .get(..16)
            .and_then(|prefix| u64::from_str_radix(prefix, 16).ok())
            .unwrap_or_default()
    }
}

impl fmt::Display for TaskKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Hashes `value` as JSON, prefixed by a domain so different kinds of key never collide.
///
/// Key material must serialize deterministically: structs and `BTreeMap`s only, no hash maps.
pub fn stable_digest(domain: &str, value: &impl Serialize) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0u8]);
    hasher.update(serde_json::to_vec(value)?);
    Ok(hex(&hasher.finalize()))
}

/// Hashes a file's contents.
pub fn file_digest(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}
