use std::ffi::c_void;

use duckdb::core::FlatVector;
use vortex::aliases::hash_map::HashMap;
use vortex_duckdb::ConversionCache;

pub struct FFIConversionCache {
    inner: *mut c_void,
}

impl From<ConversionCache> for FFIConversionCache {
    fn from(buffer: ConversionCache) -> Self {
        let ptr = Box::into_raw(buffer.values_cache) as *mut c_void;
        FFIConversionCache { inner: ptr }
    }
}

impl From<FFIConversionCache> for ConversionCache {
    fn from(buffer: FFIConversionCache) -> Self {
        let inner: Box<HashMap<usize, FlatVector>> = unsafe { Box::from_raw(buffer.inner.cast()) };
        ConversionCache {
            values_cache: inner,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ConversionCache_create() -> *mut FFIConversionCache {
    let cache = ConversionCache::default();
    let cache: FFIConversionCache = cache.into();
    Box::into_raw(Box::new(cache))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ConversionCache_free(buffer: *mut FFIConversionCache) {
    let internal: Box<ConversionCache> = unsafe { Box::from_raw(buffer.cast()) };
    drop(internal)
}
