#[macro_export]
macro_rules! match_each_decimal_value {
    ($self:expr, | $_:tt $value:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $value:ident ) => ( $($body)* )}
        macro_rules! __with__ {( $_ $value:ident ) => ( $($body)* )}
        match $self {
            DecimalValue::I8(v) => __with__! { v },
            DecimalValue::I16(v) => __with__! { v },
            DecimalValue::I32(v) => __with__! { v },
            DecimalValue::I64(v) => __with__! { v },
            DecimalValue::I128(v) => __with__! { v },
            DecimalValue::I256(v) => __with__! { v },
        }
    });
}

/// Macro to match over each decimal value type, binding the corresponding native type (from `DecimalValueType`)
#[macro_export]
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::arrays::DecimalValueType;
        use vortex_scalar::i256;
        match $self {
            DecimalValueType::I8 => __with__! { i8 },
            DecimalValueType::I16 => __with__! { i16 },
            DecimalValueType::I32 => __with__! { i32 },
            DecimalValueType::I64 => __with__! { i64 },
            DecimalValueType::I128 => __with__! { i128 },
            DecimalValueType::I256 => __with__! { i256 },
        }
    });
    ($self:expr, | ($_0:tt $enc:ident, $_1:tt $dv_path:ident) | $($body:tt)*) => ({
        macro_rules! __with2__ { ( $_0 $enc:ident, $_1 $dv_path:ident ) => ( $($body)* ) }
        use $crate::arrays::DecimalValueType;
        use vortex_scalar::i256;
        use vortex_scalar::DecimalValue::*;

        match $self {
            DecimalValueType::I8 => __with2__! { i8, I8 },
            DecimalValueType::I16 => __with2__! { i16, I16 },
            DecimalValueType::I32 => __with2__! { i32, I32 },
            DecimalValueType::I64 => __with2__! { i64, I64 },
            DecimalValueType::I128 => __with2__! { i128, I128 },
            DecimalValueType::I256 => __with2__! { i256, I256 },
        }
    });
    ($self:expr, $todo:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::arrays::DecimalValueType;
        use vortex_scalar::i256;
        match $self {
            DecimalValueType::I8 => __with__! { i8 },
            DecimalValueType::I16 => __with__! { i16 },
            DecimalValueType::I32 => __with__! { i32 },
            DecimalValueType::I64 => __with__! { i64 },
            DecimalValueType::I128 => __with__! { i128 },
            DecimalValueType::I256 => __with__! { i256 },
            _ => $todo
        }
    });
}
