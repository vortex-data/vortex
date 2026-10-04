// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting a plan's tasks across machines or processes.

use std::fmt;
use std::str::FromStr;

use anyhow::Context;
use anyhow::ensure;

use crate::key::TaskKey;

/// Shard `index` of `count`. A task belongs to exactly one shard, chosen from its key, so the
/// split is stable across runs and independent of task order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    index: u64,
    count: u64,
}

impl Shard {
    /// Creates shard `index` of `count`.
    pub fn new(index: u64, count: u64) -> anyhow::Result<Self> {
        ensure!(count > 0, "shard count must be greater than zero");
        ensure!(index < count, "shard index {index} must be less than the count {count}");
        Ok(Self { index, count })
    }

    /// Whether the task with this key belongs to this shard.
    pub fn contains(&self, key: &TaskKey) -> bool {
        key.bucket() % self.count == self.index
    }
}

impl FromStr for Shard {
    type Err = anyhow::Error;

    /// Parses `index/count`, e.g. `3/16`.
    fn from_str(text: &str) -> anyhow::Result<Self> {
        let (index, count) = text
            .split_once('/')
            .with_context(|| format!("shard `{text}` must look like index/count, e.g. 3/16"))?;
        Self::new(
            index.trim().parse().context("parsing shard index")?,
            count.trim().parse().context("parsing shard count")?,
        )
    }
}

impl fmt::Display for Shard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.index, self.count)
    }
}
