use std::ffi::{c_uint, c_void};

use vortex_duckdb::ConversionCache;

pub struct VXConversionCache {
    pub inner: *mut c_void,
}

impl From<Box<ConversionCache>> for VXConversionCache {
    fn from(buffer: Box<ConversionCache>) -> Self {
        let ptr = Box::into_raw(buffer) as *mut c_void;
        VXConversionCache { inner: ptr }
    }
}

pub unsafe fn into_conversion_cache<'a>(cache: *mut VXConversionCache) -> &'a mut ConversionCache {
    unsafe { &mut *(*cache).inner.cast() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_conversion_cache_create(id: c_uint) -> *mut VXConversionCache {
    let cache: VXConversionCache = Box::new(ConversionCache::new(id as u64)).into();
    Box::into_raw(Box::new(cache))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_conversion_cache_free(buffer: *mut VXConversionCache) {
    let internal: Box<ConversionCache> =
        unsafe { Box::from_raw(Box::from_raw(buffer).inner.cast()) };
    drop(internal)
}
