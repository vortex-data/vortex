/// Create a new struct type. For example:
///
/// ```
/// use vortex_dtype::{struct_type, DType, PType};
///
/// let the_struct = struct_type! {
///     "x" => DType::from(PType::F64),
///     "y" => DType::from(PType::F64),
/// };
///
/// assert!(the_struct.is_struct());
/// assert_eq!(the_struct.as_struct().unwrap().nfields(), 2);
/// ```
///
/// By default, the returned struct DType does not support top-level nulls.
/// However, you can override this:
///
/// ```
/// use vortex_dtype::{struct_type, DType, PType};
///
/// let the_struct = struct_type! { nullable;
///     "x" => DType::from(PType::F64),
///     "y" => DType::from(PType::F64),
/// };
///
/// assert!(the_struct.is_struct());
/// assert!(the_struct.is_nullable());
/// assert_eq!(the_struct.as_struct().unwrap().nfields(), 2);
/// ```
///
#[macro_export]
macro_rules! struct_type {
    ($($name:expr => $dtype:expr),* $(,)?) => {{
        $crate::DType::Struct(::std::sync::Arc::new($crate::StructDType::from_iter([
            $(($name , $dtype)),*
        ])), $crate::Nullability::NonNullable)
    }};
    (nullable; $($name:expr => $dtype:expr),* $(,)?) => {{
        $crate::DType::Struct(::std::sync::Arc::new($crate::StructDType::from_iter([
            $(($name , $dtype)),*
        ])), $crate::Nullability::Nullable)
    }};
}

#[cfg(test)]
mod tests {
    use crate::{DType, PType};

    #[test]
    fn test_macros() {
        let struct_type = struct_type! {
            "x" => DType::from(PType::F64),
            "y" => DType::from(PType::F64),
        };
        assert!(struct_type.is_struct());
        assert!(!struct_type.is_nullable());
        assert_eq!(struct_type.as_struct().unwrap().nfields(), 2);

        let nullable_struct = struct_type! { nullable;
            "x" => DType::from(PType::F64),
            "y" => DType::from(PType::F64),
        };
        assert!(nullable_struct.is_struct());
        assert!(nullable_struct.is_nullable());
        assert_eq!(nullable_struct.as_struct().unwrap().nfields(), 2);
    }
}
