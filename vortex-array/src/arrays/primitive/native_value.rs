use vortex_dtype::{NativePType, half};

/// NativeValue serves as a wrapper type to allow us to implement Hash and Eq on all primitive types.
///
/// Rust does not define Hash/Eq for any of the float types due to the presence of
/// NaN and +/- 0. We don't care about storing multiple NaNs or zeros in our dictionaries,
/// so we define simple bit-wise Hash/Eq for the Value-wrapped versions of these types.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct NativeValue<T>(pub T);

impl<T: NativePType> PartialEq<NativeValue<T>> for NativeValue<T> {
    fn eq(&self, other: &NativeValue<T>) -> bool {
        self.0.is_eq(other.0)
    }
}

impl<T: NativePType> Eq for NativeValue<T> {}

macro_rules! prim_value {
    ($typ:ty) => {
        impl core::hash::Hash for NativeValue<$typ> {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
}

macro_rules! float_value {
    ($typ:ty) => {
        impl core::hash::Hash for NativeValue<$typ> {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.0.to_bits().hash(state);
            }
        }
    };
}

prim_value!(u8);
prim_value!(u16);
prim_value!(u32);
prim_value!(u64);
prim_value!(i8);
prim_value!(i16);
prim_value!(i32);
prim_value!(i64);
float_value!(half::f16);
float_value!(f32);
float_value!(f64);
