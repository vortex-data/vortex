use std::ffi::{c_uint, c_void};

use vortex_duckdb::ConversionCache;

pub struct FFIConversionCache {
    pub inner: *mut c_void,
}

impl From<Box<ConversionCache>> for FFIConversionCache {
    fn from(buffer: Box<ConversionCache>) -> Self {
        let ptr = Box::into_raw(buffer) as *mut c_void;
        FFIConversionCache { inner: ptr }
    }
}

pub unsafe fn into_conversion_cache<'a>(cache: *mut FFIConversionCache) -> &'a mut ConversionCache {
    unsafe { &mut *(*cache).inner.cast() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ConversionCache_create(id: c_uint) -> *mut FFIConversionCache {
    let cache: FFIConversionCache = Box::new(ConversionCache::new(id as u64)).into();
    Box::into_raw(Box::new(cache))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ConversionCache_free(buffer: *mut FFIConversionCache) {
    let internal: Box<ConversionCache> = unsafe { Box::from_raw((*buffer).inner.cast()) };
    drop(internal)
}
