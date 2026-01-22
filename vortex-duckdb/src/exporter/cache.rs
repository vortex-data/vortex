// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex_utils::aliases::dash_map::DashMap;

use crate::duckdb::Vector;

/// Cache for array conversions from Vortex to DuckDB.
///
/// Uses the memory address of `ArrayRef` pointers as cache keys
/// to avoid redundant conversions of the same array instances.
/// We hold on to the `ArrayRef` to ensure that the key (ptr addr) doesn't get reused.
#[derive(Default)]
pub struct ConversionCache {
    pub values_cache: DashMap<usize, (ArrayRef, Arc<Mutex<Vector>>)>,
    pub canonical_cache: DashMap<usize, (ArrayRef, Canonical)>,
    // A value which must be unique for a given DuckDB operator.
    instance_id: u64,
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
