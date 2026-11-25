// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use num_traits::Float;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_float_ptype;
use crate::proto::expr::CompareMode as PbCompareMode;
use crate::proto::expr::boolean_opts::BooleanOp;
use crate::proto::expr::compare_mode::FloatOrder as PbFloatOrder;
use crate::proto::expr::compare_mode::TypeOrder as PbTypeOrder;
use crate::proto::expr::compare_mode::type_order::Order as PbOrder;
use crate::proto::expr::compare_opts::CompareOp;
use crate::scalar::NumericOperator;
use crate::scalar::PValue;
use crate::scalar::Scalar;

/// The two boolean operators.
///
/// They obey [Kleene (three-valued) logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics),
/// so they are not strict: `false AND null` is `false`, and `true OR null` is `true`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BooleanOperator {
    /// Boolean AND (∧).
    And,
    /// Boolean OR (∨).
    Or,
}

impl Display for BooleanOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            BooleanOperator::And => "and",
            BooleanOperator::Or => "or",
        };
        Display::fmt(display, f)
    }
}

impl From<BooleanOperator> for BooleanOp {
    fn from(value: BooleanOperator) -> Self {
        match value {
            BooleanOperator::And => BooleanOp::And,
            BooleanOperator::Or => BooleanOp::Or,
        }
    }
}

impl From<BooleanOp> for BooleanOperator {
    fn from(value: BooleanOp) -> Self {
        match value {
            BooleanOp::And => BooleanOperator::And,
            BooleanOp::Or => BooleanOperator::Or,
        }
    }
}

/// The six comparison operators, providing compile-time guarantees that only
/// comparison variants are used where comparisons are expected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareOperator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another.
    Gt,
    /// Expression is greater or equal to another.
    Gte,
    /// Expression is less than another.
    Lt,
    /// Expression is less or equal to another.
    Lte,
}

impl CompareOperator {
    /// Return the logical inverse of this comparison operator.
    pub fn inverse(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::NotEq,
            CompareOperator::NotEq => CompareOperator::Eq,
            CompareOperator::Gt => CompareOperator::Lte,
            CompareOperator::Gte => CompareOperator::Lt,
            CompareOperator::Lt => CompareOperator::Gte,
            CompareOperator::Lte => CompareOperator::Gt,
        }
    }

    /// Swap the sides of the operator so that swapping lhs and rhs preserves the result.
    pub fn swap(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::Eq,
            CompareOperator::NotEq => CompareOperator::NotEq,
            CompareOperator::Gt => CompareOperator::Lt,
            CompareOperator::Gte => CompareOperator::Lte,
            CompareOperator::Lt => CompareOperator::Gt,
            CompareOperator::Lte => CompareOperator::Gte,
        }
    }

    /// Returns the total-order comparison that this operator applies to two native values.
    pub fn to_predicate<T: NativePType>(self) -> fn(T, T) -> bool {
        match self {
            CompareOperator::Eq => T::is_eq,
            CompareOperator::NotEq => |lhs, rhs| !lhs.is_eq(rhs),
            CompareOperator::Gt => T::is_gt,
            CompareOperator::Gte => T::is_ge,
            CompareOperator::Lt => T::is_lt,
            CompareOperator::Lte => T::is_le,
        }
    }
}

/// How a comparison orders the signed zeros and the `NaN` values of a float.
///
/// The enum is open: a later version can add an order. It has no default: each [`CompareMode`]
/// names its order of floats.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FloatOrder {
    /// `-0.0 = +0.0`. `NaN` values are ordered by sign and payload, as in [`f64::total_cmp`]: a
    /// `NaN` with the sign bit set is less than `-inf`, a `NaN` with the sign bit clear is
    /// greater than `+inf`, and two `NaN` values are equal only when their bits are equal.
    ///
    /// DataFusion uses this order.
    ZerosEqual,
    /// The total order over the bit patterns of floats, as in [`f64::total_cmp`]: `-0.0 < +0.0`,
    /// and `NaN` values are ordered as in [`FloatOrder::ZerosEqual`]. Equality is bitwise.
    ///
    /// This is the order of [`Scalar`] values and of Vortex statistics. Code that must keep each
    /// bit pattern, such as compression, uses this order.
    TotalOrder,
    /// `-0.0 = +0.0`. All `NaN` values are equal to each other and greater than all other values,
    /// whatever their sign and payload. Spark, DuckDB, and Polars use this order.
    Canonical,
    /// `-0.0 = +0.0`. `NaN` is unordered, as in IEEE 754: each comparison with `NaN` is false,
    /// except `!=`, which is true. PyArrow uses this order.
    Ieee754,
}

impl From<FloatOrder> for PbFloatOrder {
    fn from(value: FloatOrder) -> Self {
        match value {
            FloatOrder::ZerosEqual => Self::ZerosEqual,
            FloatOrder::TotalOrder => Self::TotalOrder,
            FloatOrder::Canonical => Self::Canonical,
            FloatOrder::Ieee754 => Self::Ieee754,
        }
    }
}

impl From<PbFloatOrder> for FloatOrder {
    fn from(value: PbFloatOrder) -> Self {
        match value {
            PbFloatOrder::ZerosEqual => Self::ZerosEqual,
            PbFloatOrder::TotalOrder => Self::TotalOrder,
            PbFloatOrder::Canonical => Self::Canonical,
            PbFloatOrder::Ieee754 => Self::Ieee754,
        }
    }
}

impl Display for FloatOrder {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            FloatOrder::ZerosEqual => "zeros_equal",
            FloatOrder::TotalOrder => "total_order",
            FloatOrder::Canonical => "canonical",
            FloatOrder::Ieee754 => "ieee754",
        };

        Display::fmt(display, f)
    }
}

/// How a type-specific comparison orders the values in its domain.
///
/// A mode has one order for each type whose values have more than one useful order. A type
/// without an order in the mode has only one order, and compares the same in each mode.
///
/// | Type   | Order          |
/// |--------|----------------|
/// | Floats | [`FloatOrder`] |
///
/// Integer, decimal, boolean, and byte comparisons give the same result in each mode.
///
/// A mode has no default: [`CompareMode::from_float_order`] takes the order of floats.
/// [`CompareMode::new`], which selects [`FloatOrder::ZerosEqual`], is deprecated. The functions
/// that take no mode, such as `eq` and `ArrayBuiltins::compare`, still use it.
///
/// To add the order of another type, add a field, a `with_*` function, and its part of
/// [`CompareMode::for_dtype`] and [`CompareMode::agrees_with_total_order`].
///
/// | Engine                 | [`FloatOrder`] |
/// |------------------------|----------------|
/// | Functions without mode | `ZerosEqual`   |
/// | DataFusion             | `ZerosEqual`   |
/// | Spark, DuckDB, Polars  | `Canonical`    |
/// | PyArrow                | `Ieee754`      |
///
/// Parquet has no comparison semantics: it defines an order only for its statistics. Its
/// `IEEE_754_TOTAL_ORDER` column order is [`FloatOrder::TotalOrder`]. Its older `TYPE_ORDER`
/// leaves `NaN` and the signed zeros ambiguous, so a reader must take a minimum of `+0.0` and a
/// maximum of `-0.0` to include the other zero, and must ignore a `NaN` bound.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompareMode {
    float_order: FloatOrder,
}

impl CompareMode {
    /// The total order of each type. For floats, this is the order over the bit patterns, as in
    /// [`f64::total_cmp`].
    ///
    /// This is the order of [`Scalar`] values and of Vortex statistics. Use it when the
    /// comparison must keep `-0.0` and `+0.0` distinct.
    pub const TOTAL_ORDER: Self = Self::from_float_order(FloatOrder::TotalOrder);

    /// Returns the mode that orders floats in `float_order`.
    pub const fn from_float_order(float_order: FloatOrder) -> Self {
        Self { float_order }
    }

    /// Returns the mode that orders floats in [`FloatOrder::ZerosEqual`]: `-0.0 = +0.0`, and all
    /// other floats compare in total order.
    ///
    /// The functions that take no mode compare in this mode.
    #[deprecated(note = "name the order of floats: use `CompareMode::from_float_order`")]
    // A mode has no default, so there is no `Default` implementation.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self::from_float_order(FloatOrder::ZerosEqual)
    }

    /// Returns this mode with the given order of floats.
    pub const fn with_float_order(mut self, float_order: FloatOrder) -> Self {
        self.float_order = float_order;
        self
    }

    /// Returns how the mode orders floats.
    pub fn float_order(self) -> FloatOrder {
        self.float_order
    }

    /// Returns whether this mode is [`CompareMode::TOTAL_ORDER`].
    pub fn is_total_order(self) -> bool {
        self == Self::TOTAL_ORDER
    }

    /// Returns whether this mode orders floats in [`FloatOrder::ZerosEqual`], as the functions
    /// that take no mode do. A display of the comparison then leaves the mode out.
    pub fn is_zeros_equal(self) -> bool {
        self.float_order == FloatOrder::ZerosEqual
    }

    /// Returns this mode with one fixed order for each type that values of `dtype` do not
    /// contain, because that order cannot change their comparison. Equal comparisons are then
    /// equal expressions.
    ///
    /// For example, the order of floats is [`FloatOrder::ZerosEqual`] when `dtype` contains no
    /// floats.
    pub fn for_dtype(self, dtype: &DType) -> Self {
        let mut mode = self;

        if !contains_float(dtype) {
            mode.float_order = FloatOrder::ZerosEqual;
        }

        mode
    }

    /// Returns whether comparing values of `dtype` with `operator` in this mode always gives the
    /// same result as in [`CompareMode::TOTAL_ORDER`].
    ///
    /// `constant` is the right-hand side when it is a known constant. Total-order kernels,
    /// statistics, and indexes can evaluate or prune a comparison that agrees.
    ///
    /// The mode agrees when the order of each type agrees. For floats:
    ///
    /// - Values that contain no floats agree.
    /// - With an order where `-0.0 = +0.0`, the constant must not be zero, so that both zeros are
    ///   on the same side of it.
    /// - With [`FloatOrder::Canonical`] or [`FloatOrder::Ieee754`], the operator must also be `=`
    ///   or `!=` and the constant must not be `NaN`. A `NaN` on the left is then unequal in each
    ///   mode.
    pub fn agrees_with_total_order(
        self,
        operator: CompareOperator,
        dtype: &DType,
        constant: Option<&Scalar>,
    ) -> bool {
        self.float_order_agrees_with_total_order(operator, dtype, constant)
    }

    /// The part of [`CompareMode::agrees_with_total_order`] for the order of floats.
    fn float_order_agrees_with_total_order(
        self,
        operator: CompareOperator,
        dtype: &DType,
        constant: Option<&Scalar>,
    ) -> bool {
        if self.float_order == FloatOrder::TotalOrder || !contains_float(dtype) {
            return true;
        }

        let Some(value) = constant.and_then(float_value) else {
            return false;
        };

        if value == 0.0 {
            return false;
        }

        match self.float_order {
            FloatOrder::TotalOrder | FloatOrder::ZerosEqual => true,
            FloatOrder::Canonical | FloatOrder::Ieee754 => {
                matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) && !value.is_nan()
            }
        }
    }
}

impl CompareMode {
    /// Returns the zero `z` for which `x <operator> z` in [`CompareMode::TOTAL_ORDER`] gives the
    /// same result as `x <operator> constant` in this mode, when `constant` is a float zero.
    ///
    /// In [`FloatOrder::ZerosEqual`], an ordering comparison with a zero has such a zero: for
    /// example, `x > 0.0` is `x > +0.0` in total order, and `x >= 0.0` is `x >= -0.0`. Total-order
    /// kernels can then evaluate the comparison. Equality with a zero has no such zero, because it
    /// must match two bit patterns.
    pub fn total_order_zero(
        self,
        operator: CompareOperator,
        constant: Option<&Scalar>,
    ) -> Option<Scalar> {
        if self.float_order != FloatOrder::ZerosEqual {
            return None;
        }

        let constant = constant?;
        let DType::Primitive(ptype, nullability) = constant.dtype() else {
            return None;
        };

        if float_value(constant)? != 0.0 {
            return None;
        }

        // The zero that has both zeros on the side of it that the operator selects.
        let negative = match operator {
            CompareOperator::Gte | CompareOperator::Lt => true,
            CompareOperator::Gt | CompareOperator::Lte => false,
            CompareOperator::Eq | CompareOperator::NotEq => return None,
        };

        Some(match_each_float_ptype!(*ptype, |T| {
            signed_zero::<T>(negative, *nullability)
        }))
    }
}

/// Returns `-0.0` or `+0.0` as a scalar of the float type `T`.
fn signed_zero<T: NativePType + Float + Into<PValue>>(
    negative: bool,
    nullability: Nullability,
) -> Scalar {
    let zero = if negative { T::neg_zero() } else { T::zero() };

    Scalar::primitive(zero, nullability)
}

impl From<CompareMode> for PbCompareMode {
    fn from(value: CompareMode) -> Self {
        let float_order = PbTypeOrder {
            order: Some(PbOrder::Float(PbFloatOrder::from(value.float_order).into())),
        };

        Self {
            orders: vec![float_order],
        }
    }
}

impl TryFrom<PbCompareMode> for CompareMode {
    type Error = VortexError;

    /// Fails when the mode has an order that this version does not know, two orders of one type,
    /// or no order of floats.
    ///
    /// A later version can add the order of another type. This version decodes that order as an
    /// entry without a value and fails, because a comparison in another order gives wrong rows.
    fn try_from(value: PbCompareMode) -> VortexResult<Self> {
        let mut float_order = None;

        for type_order in value.orders {
            match type_order.order {
                Some(PbOrder::Float(code)) => {
                    let order = PbFloatOrder::try_from(code)
                        .map_err(|_| vortex_err!("unknown float order code: {code}"))?;

                    if float_order.replace(FloatOrder::from(order)).is_some() {
                        vortex_bail!("compare mode has more than one float order");
                    }
                }
                None => vortex_bail!("compare mode has the order of an unknown type"),
            }
        }

        let Some(float_order) = float_order else {
            vortex_bail!("compare mode has no float order");
        };

        Ok(Self::from_float_order(float_order))
    }
}

impl Display for CompareMode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // One `type=order` part for each type, separated by a comma.
        write!(f, "floats={}", self.float_order)
    }
}

/// Returns whether values of `dtype` contain floats, so that their comparison depends on the
/// [`CompareMode`].
pub fn contains_float(dtype: &DType) -> bool {
    match dtype {
        DType::Primitive(ptype, _) => ptype.is_float(),
        DType::Extension(ext) => contains_float(ext.storage_dtype()),
        DType::List(element, _) | DType::FixedSizeList(element, ..) => contains_float(element),
        DType::Struct(fields, _) => fields.fields().any(|field| contains_float(&field)),
        DType::Map(map, _) => {
            contains_float(&map.key_dtype()) || contains_float(&map.value_dtype())
        }
        DType::Null
        | DType::Bool(_)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Union(..)
        | DType::Variant(_) => false,
    }
}

/// Returns the value of a non-null float scalar, or of an extension scalar with float storage, as
/// an `f64`.
///
/// The conversion keeps zeros and `NaN` values, so it can classify the scalar.
fn float_value(scalar: &Scalar) -> Option<f64> {
    match scalar.dtype() {
        DType::Primitive(ptype, _) if ptype.is_float() => {
            scalar.as_primitive().pvalue()?.cast::<f64>().ok()
        }
        DType::Extension(_) => float_value(&scalar.as_extension().to_storage_scalar()),
        _ => None,
    }
}

impl Display for CompareOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            CompareOperator::Eq => "=",
            CompareOperator::NotEq => "!=",
            CompareOperator::Gt => ">",
            CompareOperator::Gte => ">=",
            CompareOperator::Lt => "<",
            CompareOperator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl From<CompareOperator> for CompareOp {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => CompareOp::Eq,
            CompareOperator::NotEq => CompareOp::NotEq,
            CompareOperator::Gt => CompareOp::Gt,
            CompareOperator::Gte => CompareOp::Gte,
            CompareOperator::Lt => CompareOp::Lt,
            CompareOperator::Lte => CompareOp::Lte,
        }
    }
}

impl From<CompareOp> for CompareOperator {
    fn from(value: CompareOp) -> Self {
        match value {
            CompareOp::Eq => CompareOperator::Eq,
            CompareOp::NotEq => CompareOperator::NotEq,
            CompareOp::Gt => CompareOperator::Gt,
            CompareOp::Gte => CompareOperator::Gte,
            CompareOp::Lt => CompareOperator::Lt,
            CompareOp::Lte => CompareOperator::Lte,
        }
    }
}

/// An operator of one of the two-operand functions [`Compare`], [`Boolean`], and [`Numeric`].
///
/// Use it where one value can hold any infix operator, such as when a binding or a query engine
/// converts its own operator codes.
///
/// [`Compare`]: crate::scalar_fn::builtin::compare::Compare
/// [`Boolean`]: crate::scalar_fn::builtin::boolean::Boolean
/// [`Numeric`]: crate::scalar_fn::builtin::numeric::Numeric
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BinaryOperator {
    /// A comparison operator.
    Compare(CompareOperator),
    /// A Kleene boolean operator.
    Boolean(BooleanOperator),
    /// An arithmetic operator.
    Numeric(NumericOperator),
}

impl From<CompareOperator> for BinaryOperator {
    fn from(operator: CompareOperator) -> Self {
        BinaryOperator::Compare(operator)
    }
}

impl From<BooleanOperator> for BinaryOperator {
    fn from(operator: BooleanOperator) -> Self {
        BinaryOperator::Boolean(operator)
    }
}

impl From<NumericOperator> for BinaryOperator {
    fn from(operator: NumericOperator) -> Self {
        BinaryOperator::Numeric(operator)
    }
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for CompareOperator {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0..=5)? {
            0 => CompareOperator::Eq,
            1 => CompareOperator::NotEq,
            2 => CompareOperator::Gt,
            3 => CompareOperator::Gte,
            4 => CompareOperator::Lt,
            5 => CompareOperator::Lte,
            _ => unreachable!(),
        })
    }
}
