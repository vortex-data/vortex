use std::sync::Arc;

use duckdb::core::FlatVector;
use vortex::aliases::hash_map::HashMap;
use vortex::error::{VortexExpect, VortexResult};
use vortex::{Array, ArrayRef, Canonical, IntoArray};

#[derive(Default)]
pub struct ConversionCache {
    pub values_cache: HashMap<usize, (ArrayRef, FlatVector)>,
    pub canonical_cache: HashMap<usize, (ArrayRef, Canonical)>,
    // A value which must be unique for a given duckdb pipeline.
    pub instance_id: u64,
}

impl ConversionCache {
    pub fn new(id: u64) -> Self {
        Self {
            instance_id: id,
            ..Self::default()
        }
    }

    fn insert_cached_array(
        &mut self,
        arr_value: usize,
        array: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
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

    pub fn cached_array(&mut self, array: &ArrayRef) -> VortexResult<ArrayRef> {
        let arr_value = Arc::as_ptr(array).addr();

        let entry = self.canonical_cache.get(&arr_value);
        match entry {
            None => self.insert_cached_array(arr_value, array),
            Some((cached_array_ref, cached_canonical)) => {
                if Arc::ptr_eq(cached_array_ref, array) {
                    Ok(cached_canonical.clone().into_array())
                } else {
                    self.insert_cached_array(arr_value, array)
                }
            }
        }
    }
}
