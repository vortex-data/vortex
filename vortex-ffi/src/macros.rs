// From hyper (MIT licensed): https://github.com/hyperium/hyper/blob/master/src/ffi/macros.rs
macro_rules! non_null {
    ($ptr:ident, $eval:expr, $err:expr) => {{
        if $ptr.is_null() {
            return $err;
        }
        unsafe { $eval }
    }};
    (& * $ptr:ident,returning: $err:expr) => {{ non_null!($ptr, &*$ptr, $err) }};
    (&mut * $ptr:ident,returning: $err:expr) => {{ non_null!($ptr, &mut *$ptr, $err) }};
    (Box::from_raw($ptr:ident),returning: $err:expr) => {{ non_null!($ptr, Box::from_raw($ptr), $err) }};
    (Arc::from_raw($ptr:ident),returning: $err:expr) => {{ non_null!($ptr, Arc::from_raw($ptr), $err) }};
}
