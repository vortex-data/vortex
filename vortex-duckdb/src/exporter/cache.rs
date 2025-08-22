// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use dashmap::DashMap;
use parking_lot::Mutex;
use std::fmt::Debug;
use std::sync::Arc;
use vortex::ArrayRef;

use crate::duckdb::Vector;

/// Cache for array conversions from Vortex to DuckDB.
///
/// Uses the memory address of `ArrayRef` pointers as cache keys
/// to avoid redundant conversions of the same array instances.
#[derive(Default)]
pub struct ConversionCache {
    pub values_cache: DashMap<usize, (ArrayRef, Arc<Mutex<Vector>>)>,
    // A value which must be unique for a given DuckDB pipeline.
    instance_id: u64,
}

impl Debug for ConversionCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversionCache")
            .field("values_cache_len", &self.values_cache.len())
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl ConversionCache {
    pub fn new(id: u64) -> Self {
        Self {
            instance_id: id,
            ..Self::default()
        }
    }

    pub fn instance_id(&self) -> u64 {
        self.instance_id
    }
}
