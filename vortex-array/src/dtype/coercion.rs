// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities for performing type coercion.

use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::decimal::DecimalDType;

impl PType {
    /// Returns the least supertype (widest common type) of two primitive types,
    /// or `None` if no lossless promotion exists.
    pub fn least_supertype(self, other: PType) -> Option<PType> {
        if self == other {
            return Some(self);
        }

        // Same family — pick the wider.
        if self.is_unsigned_int() && other.is_unsigned_int() {
            return Some(self.max_unsigned_ptype(other));
        }
        if self.is_signed_int() && other.is_signed_int() {
            return Some(self.max_signed_ptype(other));
        }
        if self.is_float() && other.is_float() {
            return if self.byte_width() >= other.byte_width() {
                Some(self)
            } else {
                Some(other)
            };
        }

        // Unsigned + Signed crossover — promote to signed one width-step wider.
        if self.is_unsigned_int() && other.is_signed_int() {
            return Self::unsigned_signed_supertype(self, other);
        }
        if self.is_signed_int() && other.is_unsigned_int() {
            return Self::unsigned_signed_supertype(other, self);
        }

        // Int + Float — pick the smallest float that losslessly represents the integer.
        let (int, float) = if self.is_float() {
            (other, self)
        } else {
            (self, other)
        };
        Self::int_float_supertype(int, float)
    }

    fn unsigned_signed_supertype(unsigned: PType, signed: PType) -> Option<PType> {
        use PType::*;
        match unsigned.byte_width().max(signed.byte_width()) {
            1 => Some(I16),
            2 => Some(I32),
            4 => Some(I64),
            _ => None, // U64 + I64 — no lossless 128-bit integer type
        }
    }

    fn int_float_supertype(int: PType, float: PType) -> Option<PType> {
        use PType::*;
        let min_float = match int.byte_width() {
            1 => F16,         // f16 has 11-bit mantissa, enough for 8-bit ints
            2 => F32,         // f32 has 24-bit mantissa, enough for 16-bit ints
            4 => F64,         // f64 has 53-bit mantissa, enough for 32-bit ints
            _ => return None, // no standard float for 64-bit ints
        };
        if float.byte_width() >= min_float.byte_width() {
            Some(float)
        } else {
            Some(min_float)
        }
    }
}

impl DType {
    /// The core primitive — what type can hold both `self` and `other`?
    /// Returns `None` if no common supertype exists.
    pub fn least_supertype(&self, other: &DType) -> Option<DType> {
        // 1. Identity (ignoring nullability): return self with union nullability
        if self.eq_ignore_nullability(other) {
            return Some(self.with_nullability(self.nullability() | other.nullability()));
        }

        let union_null = self.nullability() | other.nullability();

        // 2. Null + X: return X as nullable
        if matches!(self, DType::Null) {
            return Some(other.as_nullable());
        }
        if matches!(other, DType::Null) {
            return Some(self.as_nullable());
        }

        // 3. Bool + numeric: return the numeric type (with union nullability)
        if self.is_boolean() && other.is_numeric() {
            return Some(other.with_nullability(union_null));
        }
        if other.is_boolean() && self.is_numeric() {
            return Some(self.with_nullability(union_null));
        }

        // 4. Primitive + Primitive (different ptypes): delegate to PType::least_supertype
        if let (DType::Primitive(lhs_p, _), DType::Primitive(rhs_p, _)) = (self, other) {
            return lhs_p
                .least_supertype(*rhs_p)
                .map(|p| DType::Primitive(p, union_null));
        }

        // 5. Decimal + Decimal: compute wider decimal
        if let (DType::Decimal(lhs_d, _), DType::Decimal(rhs_d, _)) = (self, other) {
            return decimal_least_supertype(*lhs_d, *rhs_d).map(|d| DType::Decimal(d, union_null));
        }

        // 6. Decimal + integer Primitive: convert integer to Decimal, then widen
        if let (DType::Decimal(dec, _), DType::Primitive(p, _)) = (self, other)
            && p.is_int()
        {
            let int_dec = DecimalDType::new(integer_decimal_precision(*p), 0);
            return decimal_least_supertype(*dec, int_dec).map(|d| DType::Decimal(d, union_null));
        }
        if let (DType::Primitive(p, _), DType::Decimal(dec, _)) = (self, other)
            && p.is_int()
        {
            let int_dec = DecimalDType::new(integer_decimal_precision(*p), 0);
            return decimal_least_supertype(int_dec, *dec).map(|d| DType::Decimal(d, union_null));
        }

        // 7. Extension + anything: delegate to vtable
        if let DType::Extension(ext) = self {
            return ext
                .least_supertype(other)
                .map(|dt| dt.with_nullability(union_null));
        }
        if let DType::Extension(ext) = other {
            return ext
                .least_supertype(self)
                .map(|dt| dt.with_nullability(union_null));
        }

        // 8. Everything else: no common supertype
        None
    }

    /// Fold over a slice — what type can hold all of these?
    pub fn least_supertype_of(types: &[DType]) -> Option<DType> {
        types
            .iter()
            .skip(1)
            .try_fold(types[0].clone(), |acc, t| acc.least_supertype(t))
    }

    /// Is there any implicit coercion path from `other` to `self`?
    pub fn can_coerce_from(&self, other: &DType) -> bool {
        // 1. Same type (ignoring nullability): check nullability compatibility
        if self.eq_ignore_nullability(other) {
            return self.is_nullable() || !other.is_nullable();
        }

        // 2. Null → nullable target
        if matches!(other, DType::Null) {
            return self.is_nullable();
        }

        // 3. Bool → numeric
        if other.is_boolean() && self.is_numeric() {
            return self.is_nullable() || !other.is_nullable();
        }

        // 4. Primitive widening: true if least_supertype(source, target) == target
        if let (DType::Primitive(..), DType::Primitive(..)) = (self, other) {
            return other
                .least_supertype(self)
                .is_some_and(|st| st.eq_ignore_nullability(self))
                && (self.is_nullable() || !other.is_nullable());
        }

        // 5. Decimal widening
        if let (DType::Decimal(target, _), DType::Decimal(source, _)) = (self, other) {
            let target_integral = target.precision() as i16 - target.scale() as i16;
            let source_integral = source.precision() as i16 - source.scale() as i16;
            return target_integral >= source_integral
                && target.scale() >= source.scale()
                && (self.is_nullable() || !other.is_nullable());
        }

        // 6. Integer → Decimal
        if let (DType::Decimal(dec, _), DType::Primitive(p, _)) = (self, other)
            && p.is_int()
        {
            let needed = integer_decimal_precision(*p);
            let integral_digits = dec.precision() as i16 - dec.scale() as i16;
            return integral_digits >= needed as i16
                && (self.is_nullable() || !other.is_nullable());
        }

        // 7. Extension: delegate to vtable
        if let DType::Extension(ext) = self {
            return ext.can_coerce_from(other);
        }

        // 8. Everything else: false
        false
    }

    /// Convenience — is there a path from `self` to `other`?
    pub fn can_coerce_to(&self, other: &DType) -> bool {
        if let DType::Extension(ext) = self {
            // Extension types can define coercions in either direction, so check both.
            if ext.can_coerce_to(other) {
                return true;
            }
        };

        other.can_coerce_from(self)
    }

    /// Are all types in the slice mutually coercible to a common type?
    pub fn are_coercible(types: &[DType]) -> bool {
        DType::least_supertype_of(types).is_some()
    }

    /// Can all types in the slice be coerced to a specific target?
    pub fn all_coercible_to(types: &[DType], target: &DType) -> bool {
        types.iter().all(|t| target.can_coerce_from(t))
    }

    /// Coerce a slice to a specific target — returns the vec of targets
    /// if all are coercible, `None` if any are not.
    pub fn coerce_all_to(types: &[DType], target: &DType) -> Option<Vec<DType>> {
        types
            .iter()
            .all(|t| target.can_coerce_from(t))
            .then(|| vec![target.clone(); types.len()])
    }

    /// Coerce a slice to their mutual least supertype.
    pub fn coerce_to_supertype(types: &[DType]) -> Option<Vec<DType>> {
        let supertype = DType::least_supertype_of(types)?;
        Some(vec![supertype; types.len()])
    }

    /// Is this a numeric type (primitive int/float or decimal)?
    pub fn is_numeric(&self) -> bool {
        matches!(self, DType::Primitive(..) | DType::Decimal(..))
    }

    /// Is this a temporal type (date, time, timestamp, duration)?
    pub fn is_temporal(&self) -> bool {
        match self {
            DType::Extension(ext) => {
                use crate::dtype::extension::Matcher;
                use crate::extension::datetime::AnyTemporal;
                AnyTemporal::matches(ext)
            }
            _ => false,
        }
    }
}

/// Maps integer PType widths to the minimum decimal precision needed.
fn integer_decimal_precision(ptype: PType) -> u8 {
    match ptype {
        PType::U8 | PType::I8 => 3,
        PType::U16 | PType::I16 => 5,
        PType::U32 | PType::I32 => 10,
        PType::U64 | PType::I64 => 19,
        _ => 19,
    }
}

/// Compute the least supertype of two decimal types using SQL-standard rules.
fn decimal_least_supertype(a: DecimalDType, b: DecimalDType) -> Option<DecimalDType> {
    let a_integral = a.precision() as i16 - a.scale() as i16;
    let b_integral = b.precision() as i16 - b.scale() as i16;
    let max_integral = a_integral.max(b_integral);
    let max_scale = a.scale().max(b.scale());
    let precision = u8::try_from(max_integral + max_scale as i16).ok()?;
    DecimalDType::try_new(precision, max_scale).ok()
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::PType;
    use crate::dtype::decimal::DecimalDType;
    use crate::dtype::nullability::Nullability::NonNullable;
    use crate::dtype::nullability::Nullability::Nullable;

    #[test]
    fn is_numeric() {
        assert!(DType::Primitive(PType::I32, NonNullable).is_numeric());
        assert!(DType::Primitive(PType::F64, NonNullable).is_numeric());
        assert!(DType::Decimal(DecimalDType::new(10, 2), NonNullable).is_numeric());
        assert!(!DType::Bool(NonNullable).is_numeric());
        assert!(!DType::Utf8(NonNullable).is_numeric());
        assert!(!DType::Null.is_numeric());
    }

    #[test]
    fn least_supertype_identity() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        assert_eq!(i32_nn.least_supertype(&i32_nn).unwrap(), i32_nn);
    }

    #[test]
    fn least_supertype_nullability_union() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let i32_n = DType::Primitive(PType::I32, Nullable);
        assert_eq!(i32_nn.least_supertype(&i32_n).unwrap(), i32_n);
        assert_eq!(i32_n.least_supertype(&i32_nn).unwrap(), i32_n);
    }

    #[test]
    fn least_supertype_null_absorption() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        assert_eq!(
            DType::Null.least_supertype(&i32_nn).unwrap(),
            DType::Primitive(PType::I32, Nullable)
        );
        assert_eq!(
            i32_nn.least_supertype(&DType::Null).unwrap(),
            DType::Primitive(PType::I32, Nullable)
        );
    }

    #[test]
    fn least_supertype_unsigned_widening() {
        let u8_nn = DType::Primitive(PType::U8, NonNullable);
        let u32_nn = DType::Primitive(PType::U32, NonNullable);
        assert_eq!(u8_nn.least_supertype(&u32_nn).unwrap(), u32_nn);
    }

    #[test]
    fn least_supertype_signed_widening() {
        let i16_nn = DType::Primitive(PType::I16, NonNullable);
        let i64_nn = DType::Primitive(PType::I64, NonNullable);
        assert_eq!(i16_nn.least_supertype(&i64_nn).unwrap(), i64_nn);
    }

    #[test]
    fn least_supertype_cross_family() {
        let u8_nn = DType::Primitive(PType::U8, NonNullable);
        let i8_nn = DType::Primitive(PType::I8, NonNullable);
        assert_eq!(
            u8_nn.least_supertype(&i8_nn).unwrap(),
            DType::Primitive(PType::I16, NonNullable)
        );
    }

    #[test]
    fn least_supertype_u64_i64_none() {
        let u64_nn = DType::Primitive(PType::U64, NonNullable);
        let i64_nn = DType::Primitive(PType::I64, NonNullable);
        assert!(u64_nn.least_supertype(&i64_nn).is_none());
    }

    #[test]
    fn least_supertype_int_float_promotion() {
        let u8_nn = DType::Primitive(PType::U8, NonNullable);
        let f32_nn = DType::Primitive(PType::F32, NonNullable);
        assert_eq!(u8_nn.least_supertype(&f32_nn).unwrap(), f32_nn);
    }

    #[test]
    fn least_supertype_i32_f32_to_f64() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let f32_nn = DType::Primitive(PType::F32, NonNullable);
        assert_eq!(
            i32_nn.least_supertype(&f32_nn).unwrap(),
            DType::Primitive(PType::F64, NonNullable)
        );
    }

    #[test]
    fn least_supertype_bool_numeric() {
        let bool_nn = DType::Bool(NonNullable);
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        assert_eq!(bool_nn.least_supertype(&i32_nn).unwrap(), i32_nn);
        assert_eq!(i32_nn.least_supertype(&bool_nn).unwrap(), i32_nn);
    }

    #[test]
    fn least_supertype_decimal_widening() {
        let d1 = DType::Decimal(DecimalDType::new(10, 2), NonNullable);
        let d2 = DType::Decimal(DecimalDType::new(15, 5), NonNullable);
        let result = d1.least_supertype(&d2).unwrap();
        // integral digits: max(8, 10) = 10, max scale = 5, precision = 15
        assert_eq!(
            result,
            DType::Decimal(DecimalDType::new(15, 5), NonNullable)
        );
    }

    #[test]
    fn least_supertype_incompatible_none() {
        let utf8 = DType::Utf8(NonNullable);
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        assert!(utf8.least_supertype(&i32_nn).is_none());
    }

    #[test]
    fn can_coerce_from_widening() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let i64_nn = DType::Primitive(PType::I64, NonNullable);
        assert!(i64_nn.can_coerce_from(&i32_nn));
    }

    #[test]
    fn can_coerce_from_narrowing_rejected() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let i64_nn = DType::Primitive(PType::I64, NonNullable);
        assert!(!i32_nn.can_coerce_from(&i64_nn));
    }

    #[test]
    fn can_coerce_from_nullability_constraints() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let i32_n = DType::Primitive(PType::I32, Nullable);
        assert!(i32_n.can_coerce_from(&i32_nn));
        assert!(!i32_nn.can_coerce_from(&i32_n));
    }

    #[test]
    fn can_coerce_from_null() {
        let i32_n = DType::Primitive(PType::I32, Nullable);
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        assert!(i32_n.can_coerce_from(&DType::Null));
        assert!(!i32_nn.can_coerce_from(&DType::Null));
    }

    #[test]
    fn are_coercible_mixed() {
        let types = [
            DType::Primitive(PType::I32, NonNullable),
            DType::Primitive(PType::I64, NonNullable),
        ];
        assert!(DType::are_coercible(&types));
    }

    #[test]
    fn all_coercible_to_target() {
        let types = [
            DType::Primitive(PType::I32, NonNullable),
            DType::Primitive(PType::I16, NonNullable),
        ];
        let target = DType::Primitive(PType::I64, NonNullable);
        assert!(DType::all_coercible_to(&types, &target));
    }

    #[test]
    fn coerce_to_supertype_works() {
        let types = [
            DType::Primitive(PType::U8, NonNullable),
            DType::Primitive(PType::I16, NonNullable),
        ];
        let result = DType::coerce_to_supertype(&types).unwrap();
        // U8 + I16: unsigned_signed_supertype max_width=max(1,2)=2 => I32
        assert_eq!(result, vec![DType::Primitive(PType::I32, NonNullable); 2]);
    }

    #[test]
    fn least_supertype_integer_decimal() {
        let i32_nn = DType::Primitive(PType::I32, NonNullable);
        let dec = DType::Decimal(DecimalDType::new(15, 5), NonNullable);
        let result = i32_nn.least_supertype(&dec).unwrap();
        // int_dec for I32 = Decimal(10, 0). integral digits = 10.
        // dec integral = 15 - 5 = 10.
        // max_integral = 10, max_scale = 5, precision = 15
        assert_eq!(
            result,
            DType::Decimal(DecimalDType::new(15, 5), NonNullable)
        );
    }
}
