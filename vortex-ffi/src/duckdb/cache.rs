use std::ffi::{c_uint, c_void};

use vortex_duckdb::ConversionCache;

#[allow(non_camel_case_types)]
pub struct vx_conversion_cache {
    pub inner: *mut c_void,
}

impl From<Box<ConversionCache>> for vx_conversion_cache {
    fn from(buffer: Box<ConversionCache>) -> Self {
        let ptr = Box::into_raw(buffer) as *mut c_void;
        vx_conversion_cache { inner: ptr }
    }
}

pub unsafe fn into_conversion_cache<'a>(
    cache: *mut vx_conversion_cache,
) -> &'a mut ConversionCache {
    unsafe { &mut *(*cache).inner.cast() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_conversion_cache_create(id: c_uint) -> *mut vx_conversion_cache {
    let cache: vx_conversion_cache = Box::new(ConversionCache::new(id as u64)).into();
    Box::into_raw(Box::new(cache))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_conversion_cache_free(buffer: *mut vx_conversion_cache) {
    let internal: Box<ConversionCache> =
        unsafe { Box::from_raw(Box::from_raw(buffer).inner.cast()) };
    drop(internal)
}
