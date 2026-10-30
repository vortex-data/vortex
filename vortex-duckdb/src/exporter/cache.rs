// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::error::VortexResult;
use vortex_utils::aliases::dash_map::DashMap;

use crate::duckdb::ReusableDict;
use crate::duckdb::Vector;

type DictionaryCell = Arc<OnceLock<VortexResult<ReusableDict>>>;

/// Cache for array conversions from Vortex to DuckDB.
///
/// Uses the memory address of `ArrayRef` pointers as cache keys
/// to avoid redundant conversions of the same array instances.
/// We hold on to the `ArrayRef` to ensure that the key (ptr addr) doesn't get reused.
pub struct ConversionCache {
    pub(crate) dict_cache: DashMap<usize, (ArrayRef, DictionaryCell)>,
    pub(crate) values_cache: DashMap<usize, (ArrayRef, Arc<Mutex<Vector>>)>,
    pub(crate) canonical_cache: DashMap<usize, (ArrayRef, Canonical)>,
}

impl Default for ConversionCache {
    fn default() -> Self {
        Self {
            dict_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
            values_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
            canonical_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
        }
    }
}
