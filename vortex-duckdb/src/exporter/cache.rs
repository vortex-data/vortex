// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use parking_lot::Mutex;
use std::sync::Arc;

use dashmap::DashMap;
use vortex::error::{VortexExpect, VortexResult};
use vortex::{Array, ArrayRef, Canonical, IntoArray};

use crate::duckdb::Vector;

/// Cache for array conversions from Vortex to DuckDB.
///
/// Uses the memory address of `ArrayRef` pointers as cache keys
/// to avoid redundant conversions of the same array instances.
#[derive(Default)]
pub struct ConversionCache {
    pub values_cache: DashMap<usize, (ArrayRef, Arc<Mutex<Vector>>)>,
    pub canonical_cache: DashMap<usize, (ArrayRef, Canonical)>,
    // A value which must be unique for a given DuckDB pipeline.
    pub instance_id: u64,
}

impl ConversionCache {
    pub fn new(id: u64) -> Self {
        Self {
            instance_id: id,
            ..Self::default()
        }
    }

    fn insert_cached_array(&self, arr_value: usize, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let canon = array.to_canonical()?;
        self.canonical_cache
            .insert(arr_value, (array.clone(), canon));
        Ok(self
            .canonical_cache
            .get(&arr_value)
            .vortex_expect("just added")
            .1
            .clone()
            .into_array())
    }

    pub fn cached_array(&self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let arr_value = Arc::as_ptr(array).addr();

        // Check if we have an entry and extract the information we need
        let cache_result = self.canonical_cache.get(&arr_value).map(|entry| {
            let (cached_array_ref, cached_canonical) = entry.value();
            (cached_array_ref.clone(), cached_canonical.clone())
        });

        match cache_result {
            None => self.insert_cached_array(arr_value, array),
            Some((cached_array, cached_canonical)) => {
                if Arc::ptr_eq(&cached_array, array) {
                    Ok(cached_canonical.into_array())
                } else {
                    self.insert_cached_array(arr_value, array)
                }
            }
        }
    }
}
