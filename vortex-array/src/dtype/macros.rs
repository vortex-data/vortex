// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Construct a [`DType`](crate::dtype::DType) with concise, [`Display`](std::fmt::Display)-style
/// syntax.
///
/// A trailing `?` marks the variant as [`Nullable`](crate::dtype::Nullability::Nullable);
/// omitting it produces a [`NonNullable`](crate::dtype::Nullability::NonNullable) variant. The
/// `?` can be applied to the outer type or, for recursive types, to the inner element type
/// independently.
///
/// # Variants
///
/// ## Simple
///
/// ```
/// use vortex_array::dtype;
/// use vortex_array::dtype::DType;
/// use vortex_array::dtype::Nullability::{NonNullable, Nullable};
/// use vortex_array::dtype::PType;
///
/// assert_eq!(dtype!(null), DType::Null);
/// assert_eq!(dtype!(bool), DType::Bool(NonNullable));
/// assert_eq!(dtype!(bool?), DType::Bool(Nullable));
/// assert_eq!(dtype!(i32), DType::Primitive(PType::I32, NonNullable));
/// assert_eq!(dtype!(f64?), DType::Primitive(PType::F64, Nullable));
/// assert_eq!(dtype!(utf8), DType::Utf8(NonNullable));
/// assert_eq!(dtype!(binary?), DType::Binary(Nullable));
/// ```
///
/// ## Decimal
///
/// Precision and scale must be const-evaluable; invalid values fail to compile.
///
/// ```
/// use vortex_array::dtype;
/// use vortex_array::dtype::DType;
/// use vortex_array::dtype::DecimalDType;
/// use vortex_array::dtype::Nullability::NonNullable;
///
/// assert_eq!(
///     dtype!(decimal(10, 2)),
///     DType::Decimal(DecimalDType::new(10, 2), NonNullable),
/// );
/// ```
///
/// ## Lists
///
/// ```
/// use std::sync::Arc;
///
/// use vortex_array::dtype;
/// use vortex_array::dtype::DType;
/// use vortex_array::dtype::Nullability::{NonNullable, Nullable};
/// use vortex_array::dtype::PType;
///
/// // Non-nullable list of nullable i32.
/// assert_eq!(
///     dtype!(list(i32?)),
///     DType::List(Arc::new(DType::Primitive(PType::I32, Nullable)), NonNullable),
/// );
///
/// // Fixed-size list of 16 i32s; matches the Display format.
/// assert_eq!(
///     dtype!(fixed_size_list(i32)[16]),
///     DType::FixedSizeList(Arc::new(DType::Primitive(PType::I32, NonNullable)), 16, NonNullable),
/// );
/// ```
///
/// ## Struct and Extension (no DSL — pass typed values)
///
/// `struct(expr)` accepts any expression of type [`StructFields`](crate::dtype::StructFields).
/// `extension(expr)` accepts any expression of type [`ExtDTypeRef`](crate::dtype::extension::ExtDTypeRef);
/// it carries its own nullability so no trailing `?` is accepted.
///
/// ```
/// use vortex_array::dtype;
/// use vortex_array::dtype::DType;
/// use vortex_array::dtype::Nullability::Nullable;
/// use vortex_array::dtype::StructFields;
///
/// let fields = StructFields::from_iter([("a", dtype!(i32)), ("b", dtype!(utf8?))]);
/// let dt = dtype!(struct(fields)?);
/// assert!(matches!(dt, DType::Struct(_, Nullable)));
/// ```
#[macro_export]
macro_rules! dtype {
    // Special cases that don't fit the "type + optional ?" shape:
    //   - `null` has no nullability;
    //   - `extension(ext)` carries its own nullability inside the ExtDTypeRef.
    (null) => {
        $crate::dtype::DType::Null
    };
    (extension($ext:expr)) => {
        $crate::dtype::DType::Extension($ext)
    };

    // Everything else: forward to the muncher to peel an optional trailing `?` and
    // dispatch with the resolved nullability. We can't write `($($t:tt)+ ?) => ...`
    // directly because `?` is reserved repetition metasyntax in macro patterns; a
    // hand-rolled muncher walks tokens one at a time and inspects the tail itself.
    ($($tokens:tt)+) => {
        $crate::__dtype_munch!([] $($tokens)+)
    };
}

/// Internal tt-muncher for [`dtype!`]: walks tokens into an accumulator one at a time,
/// then dispatches to [`__dtype_build!`] based on whether the input ends with `?`.
#[doc(hidden)]
#[macro_export]
macro_rules! __dtype_munch {
    // Trailing `?` after a non-empty accumulator: nullable.
    ([$($acc:tt)+] ?) => {
        $crate::__dtype_build!($crate::dtype::Nullability::Nullable, $($acc)+)
    };
    // End of input with a non-empty accumulator: non-nullable.
    ([$($acc:tt)+]) => {
        $crate::__dtype_build!($crate::dtype::Nullability::NonNullable, $($acc)+)
    };
    // Step: shift one token from the input into the accumulator.
    ([$($acc:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__dtype_munch!([$($acc)* $next] $($rest)*)
    };
}

/// Internal builder for [`dtype!`]: takes a pre-resolved [`Nullability`](crate::dtype::Nullability)
/// expression followed by the type DSL.
#[doc(hidden)]
#[macro_export]
macro_rules! __dtype_build {
    ($null:expr, bool) => {
        $crate::dtype::DType::Bool($null)
    };
    ($null:expr, utf8) => {
        $crate::dtype::DType::Utf8($null)
    };
    ($null:expr, binary) => {
        $crate::dtype::DType::Binary($null)
    };
    ($null:expr, union) => {
        $crate::dtype::DType::Union($null)
    };
    ($null:expr, variant) => {
        $crate::dtype::DType::Variant($null)
    };
    ($null:expr, decimal($p:expr, $s:expr)) => {
        $crate::dtype::DType::Decimal(
            const { $crate::dtype::DecimalDType::new_const($p, $s) },
            $null,
        )
    };
    ($null:expr, list($($inner:tt)+)) => {
        $crate::dtype::DType::List(
            ::std::sync::Arc::new($crate::dtype!($($inner)+)),
            $null,
        )
    };
    ($null:expr, fixed_size_list($($inner:tt)+)[$size:expr]) => {
        $crate::dtype::DType::FixedSizeList(
            ::std::sync::Arc::new($crate::dtype!($($inner)+)),
            $size,
            $null,
        )
    };
    ($null:expr, struct($fields:expr)) => {
        $crate::dtype::DType::Struct($fields, $null)
    };
    // Primitive catch-all: maps lowercase `u8`/`i32`/`f64`/... to `PType::U8`/`I32`/`F64`/...
    // via `paste!`. Must come last so the named variants above take priority.
    ($null:expr, $p:ident) => {
        $crate::paste::paste! {
            $crate::dtype::DType::Primitive($crate::dtype::PType::[<$p:upper>], $null)
        }
    };
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;

    #[test]
    fn null() {
        assert_eq!(dtype!(null), DType::Null);
    }

    #[test]
    fn bool() {
        assert_eq!(dtype!(bool), DType::Bool(NonNullable));
        assert_eq!(dtype!(bool?), DType::Bool(Nullable));
    }

    #[rstest]
    #[case(dtype!(u8),  PType::U8,  NonNullable)]
    #[case(dtype!(u8?), PType::U8,  Nullable)]
    #[case(dtype!(u16), PType::U16, NonNullable)]
    #[case(dtype!(u16?),PType::U16, Nullable)]
    #[case(dtype!(u32), PType::U32, NonNullable)]
    #[case(dtype!(u32?),PType::U32, Nullable)]
    #[case(dtype!(u64), PType::U64, NonNullable)]
    #[case(dtype!(u64?),PType::U64, Nullable)]
    #[case(dtype!(i8),  PType::I8,  NonNullable)]
    #[case(dtype!(i8?), PType::I8,  Nullable)]
    #[case(dtype!(i16), PType::I16, NonNullable)]
    #[case(dtype!(i16?),PType::I16, Nullable)]
    #[case(dtype!(i32), PType::I32, NonNullable)]
    #[case(dtype!(i32?),PType::I32, Nullable)]
    #[case(dtype!(i64), PType::I64, NonNullable)]
    #[case(dtype!(i64?),PType::I64, Nullable)]
    #[case(dtype!(f16), PType::F16, NonNullable)]
    #[case(dtype!(f16?),PType::F16, Nullable)]
    #[case(dtype!(f32), PType::F32, NonNullable)]
    #[case(dtype!(f32?),PType::F32, Nullable)]
    #[case(dtype!(f64), PType::F64, NonNullable)]
    #[case(dtype!(f64?),PType::F64, Nullable)]
    fn primitives(
        #[case] actual: DType,
        #[case] ptype: PType,
        #[case] nullability: crate::dtype::Nullability,
    ) {
        assert_eq!(actual, DType::Primitive(ptype, nullability));
    }

    #[test]
    fn utf8_binary() {
        assert_eq!(dtype!(utf8), DType::Utf8(NonNullable));
        assert_eq!(dtype!(utf8?), DType::Utf8(Nullable));
        assert_eq!(dtype!(binary), DType::Binary(NonNullable));
        assert_eq!(dtype!(binary?), DType::Binary(Nullable));
    }

    #[test]
    fn union_variant() {
        assert_eq!(dtype!(union), DType::Union(NonNullable));
        assert_eq!(dtype!(union?), DType::Union(Nullable));
        assert_eq!(dtype!(variant), DType::Variant(NonNullable));
        assert_eq!(dtype!(variant?), DType::Variant(Nullable));
    }

    #[test]
    fn decimal() {
        assert_eq!(
            dtype!(decimal(10, 2)),
            DType::Decimal(DecimalDType::new(10, 2), NonNullable),
        );
        assert_eq!(
            dtype!(decimal(10, 2)?),
            DType::Decimal(DecimalDType::new(10, 2), Nullable),
        );
    }

    #[test]
    fn decimal_usable_in_const_context() {
        const D: DType = dtype!(decimal(38, 10));
        assert_eq!(D, DType::Decimal(DecimalDType::new(38, 10), NonNullable));
    }

    #[test]
    fn list() {
        assert_eq!(
            dtype!(list(i32)),
            DType::List(
                Arc::new(DType::Primitive(PType::I32, NonNullable)),
                NonNullable
            ),
        );
        assert_eq!(
            dtype!(list(i32)?),
            DType::List(
                Arc::new(DType::Primitive(PType::I32, NonNullable)),
                Nullable
            ),
        );
        assert_eq!(
            dtype!(list(i32?)),
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullable)),
                NonNullable
            ),
        );
    }

    #[test]
    fn list_nested() {
        let expected = DType::List(
            Arc::new(DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullable)),
                Nullable,
            )),
            NonNullable,
        );
        assert_eq!(dtype!(list(list(i32?)?)), expected);
    }

    #[test]
    fn fixed_size_list() {
        let inner = Arc::new(DType::Primitive(PType::I32, NonNullable));
        assert_eq!(
            dtype!(fixed_size_list(i32)[16]),
            DType::FixedSizeList(Arc::clone(&inner), 16, NonNullable),
        );
        assert_eq!(
            dtype!(fixed_size_list(i32)[16]?),
            DType::FixedSizeList(inner, 16, Nullable),
        );
    }

    #[test]
    fn list_of_decimal() {
        assert_eq!(
            dtype!(list(decimal(10, 2))),
            DType::List(
                Arc::new(DType::Decimal(DecimalDType::new(10, 2), NonNullable)),
                NonNullable,
            ),
        );
    }

    #[test]
    fn r#struct() {
        let fields = StructFields::from_iter([("a", dtype!(i32)), ("b", dtype!(utf8?))]);
        let expected = DType::Struct(fields.clone(), NonNullable);
        assert_eq!(dtype!(struct(fields.clone())), expected);
        assert_eq!(
            dtype!(struct(fields)?),
            DType::Struct(expected.into_struct_fields(), Nullable),
        );
    }

    #[test]
    fn extension() {
        let ext: ExtDTypeRef = Date::new(TimeUnit::Days, NonNullable).erased();
        assert_eq!(dtype!(extension(ext.clone())), DType::Extension(ext));
    }
}
