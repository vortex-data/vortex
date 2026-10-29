// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex_utils::aliases::dash_map::DashMap;

use crate::duckdb::ReusableDict;
use crate::duckdb::Vector;

/// Cache for array conversions from Vortex to DuckDB.
///
/// Uses the memory address of `ArrayRef` pointers as cache keys
/// to avoid redundant conversions of the same array instances.
/// We hold on to the `ArrayRef` to ensure that the key (ptr addr) doesn't get reused.
pub struct ConversionCache {
    pub(super) sparse_dictionaries: bool,
    pub(crate) dict_cache: DashMap<usize, (ArrayRef, ReusableDict)>,
    pub(crate) values_cache: DashMap<usize, (ArrayRef, Arc<Mutex<Vector>>)>,
    pub(crate) canonical_cache: DashMap<usize, (ArrayRef, Canonical)>,
}

impl Default for ConversionCache {
    fn default() -> Self {
        Self {
            sparse_dictionaries: false,
            dict_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
            values_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
            canonical_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
        }
    }
}

impl ConversionCache {
    pub(crate) fn with_sparse_dictionaries() -> Self {
        Self {
            sparse_dictionaries: true,
            ..Self::default()
        }
    }
}
