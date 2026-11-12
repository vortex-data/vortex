// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexError;

use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::proto::expr::binary_opts::BinaryOp;
use crate::scalar::Scalar;

/// Equalities, inequalities, and boolean operations over possibly null values.
///
/// For most operations, if either side is null, the result is null.
///
/// The Boolean operators (And, Or) obey [Kleene (three-valued) logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Operator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another
    Gt,
    /// Expression is greater or equal to another
    Gte,
    /// Expression is less than another
    Lt,
    /// Expression is less or equal to another
    Lte,
    /// Boolean AND (∧).
    // TODO(joe): rename to KleeneAnd
    And,
    /// Boolean OR (∨).
    // TODO(joe): rename to KleeneOr
    Or,
    /// The sum of the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// The difference between the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    ///
    /// The result is null at any index that either input is null.
    Sub,
    /// Multiple two numbers
    Mul,
    /// Divide the left side by the right side
    Div,
}

impl From<Operator> for i32 {
    fn from(value: Operator) -> Self {
        let op: BinaryOp = value.into();
        op.into()
    }
}

impl From<Operator> for BinaryOp {
    fn from(value: Operator) -> Self {
        match value {
            Operator::Eq => BinaryOp::Eq,
            Operator::NotEq => BinaryOp::NotEq,
            Operator::Gt => BinaryOp::Gt,
            Operator::Gte => BinaryOp::Gte,
            Operator::Lt => BinaryOp::Lt,
            Operator::Lte => BinaryOp::Lte,
            Operator::And => BinaryOp::And,
            Operator::Or => BinaryOp::Or,
            Operator::Add => BinaryOp::Add,
            Operator::Sub => BinaryOp::Sub,
            Operator::Mul => BinaryOp::Mul,
            Operator::Div => BinaryOp::Div,
        }
    }
}

impl TryFrom<i32> for Operator {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(BinaryOp::try_from(value)?.into())
    }
}

impl From<BinaryOp> for Operator {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Eq => Operator::Eq,
            BinaryOp::NotEq => Operator::NotEq,
            BinaryOp::Gt => Operator::Gt,
            BinaryOp::Gte => Operator::Gte,
            BinaryOp::Lt => Operator::Lt,
            BinaryOp::Lte => Operator::Lte,
            BinaryOp::And => Operator::And,
            BinaryOp::Or => Operator::Or,
            BinaryOp::Add => Operator::Add,
            BinaryOp::Sub => Operator::Sub,
            BinaryOp::Mul => Operator::Mul,
            BinaryOp::Div => Operator::Div,
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match &self {
            Operator::Eq => "=",
            Operator::NotEq => "!=",
            Operator::Gt => ">",
            Operator::Gte => ">=",
            Operator::Lt => "<",
            Operator::Lte => "<=",
            Operator::And => "and",
            Operator::Or => "or",
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
        };
        Display::fmt(display, f)
    }
}

impl Operator {
    pub fn inverse(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::NotEq),
            Operator::NotEq => Some(Operator::Eq),
            Operator::Gt => Some(Operator::Lte),
            Operator::Gte => Some(Operator::Lt),
            Operator::Lt => Some(Operator::Gte),
            Operator::Lte => Some(Operator::Gt),
            Operator::And
            | Operator::Or
            | Operator::Add
            | Operator::Sub
            | Operator::Mul
            | Operator::Div => None,
        }
    }

    pub fn logical_inverse(self) -> Option<Self> {
        match self {
            Operator::And => Some(Operator::Or),
            Operator::Or => Some(Operator::And),
            _ => None,
        }
    }

    /// Change the sides of the operator, so that changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::Eq),
            Operator::NotEq => Some(Operator::NotEq),
            Operator::Gt => Some(Operator::Lt),
            Operator::Gte => Some(Operator::Lte),
            Operator::Lt => Some(Operator::Gt),
            Operator::Lte => Some(Operator::Gte),
            Operator::And => Some(Operator::And),
            Operator::Or => Some(Operator::Or),
            Operator::Add => Some(Operator::Add),
            Operator::Mul => Some(Operator::Mul),
            Operator::Sub | Operator::Div => None,
        }
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }

    pub fn is_comparison(&self) -> bool {
        self.to_compare_operator().is_some()
    }

    /// Return the [`CompareOperator`] for this operator, or `None` if it is not a comparison.
    ///
    /// Prefer this over [`CompareOperator::try_from`] when a non-comparison operator is an expected
    /// case, since it does not construct an error.
    pub fn to_compare_operator(self) -> Option<CompareOperator> {
        match self {
            Operator::Eq => Some(CompareOperator::Eq),
            Operator::NotEq => Some(CompareOperator::NotEq),
            Operator::Gt => Some(CompareOperator::Gt),
            Operator::Gte => Some(CompareOperator::Gte),
            Operator::Lt => Some(CompareOperator::Lt),
            Operator::Lte => Some(CompareOperator::Lte),
            Operator::And
            | Operator::Or
            | Operator::Add
            | Operator::Sub
            | Operator::Mul
            | Operator::Div => None,
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

/// How a comparison treats the two signed zeros of a float.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SignedZeros {
    /// `-0.0 < +0.0`. Apache Iceberg and Apache Paimon use this order.
    #[default]
    Distinct,
    /// `-0.0 = +0.0`. DataFusion, Spark, DuckDB, ClickHouse, and NumPy use this order.
    Equal,
}

/// How a comparison treats `NaN` values of a float.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NanOrder {
    /// `NaN` values are ordered by sign and payload, as in [`f64::total_cmp`]: a `NaN` with the
    /// sign bit set is less than `-inf`, and a `NaN` with the sign bit clear is greater than
    /// `+inf`. Equality is bitwise. DataFusion uses this order.
    #[default]
    TotalOrder,
    /// All `NaN` values are equal to each other and greater than all other values, whatever their
    /// sign and payload. Spark, DuckDB, Apache Iceberg, and Apache Paimon use this order.
    Canonical,
    /// `NaN` is unordered, as in IEEE 754: each comparison with `NaN` is false, except `!=`, which
    /// is true. ClickHouse and NumPy use this order.
    Ieee754,
}

/// How a comparison orders a null value inside a nested value, such as a null struct field or a
/// null list element.
///
/// A null at the top level always makes the result null. Nested nulls are equal to each other.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NestedNulls {
    /// A nested null is less than all values. This is the order of [`Scalar`] values.
    #[default]
    First,
    /// A nested null is greater than all values. DuckDB uses this order.
    Last,
}

/// How a comparison orders floating-point values and the nulls inside nested values.
///
/// The float settings change only operands that contain floats, and [`NestedNulls`] changes only
/// nested operands. Integer, decimal, boolean, and byte comparisons give the same result in each
/// mode. The default mode is [`CompareMode::TOTAL_ORDER`].
///
/// | Engine            | [`SignedZeros`] | [`NanOrder`]  |
/// |-------------------|-----------------|---------------|
/// | Vortex (default)  | `Distinct`      | `TotalOrder`  |
/// | DataFusion        | `Equal`         | `TotalOrder`  |
/// | Spark, DuckDB     | `Equal`         | `Canonical`   |
/// | Iceberg, Paimon   | `Distinct`      | `Canonical`   |
/// | ClickHouse, NumPy | `Equal`         | `Ieee754`     |
///
/// DuckDB also uses [`NestedNulls::Last`].
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompareMode {
    signed_zeros: SignedZeros,
    nan_order: NanOrder,
    nested_nulls: NestedNulls,
}

impl CompareMode {
    /// The total order over the bit patterns of floats, as in [`f64::total_cmp`].
    ///
    /// This is the order of [`Scalar`] values and of Vortex statistics.
    pub const TOTAL_ORDER: Self = Self::new(SignedZeros::Distinct, NanOrder::TotalOrder);

    /// Returns the mode with the given treatment of signed zeros and `NaN` values.
    ///
    /// Nested nulls order as in [`NestedNulls::First`].
    pub const fn new(signed_zeros: SignedZeros, nan_order: NanOrder) -> Self {
        Self {
            signed_zeros,
            nan_order,
            nested_nulls: NestedNulls::First,
        }
    }

    /// Returns this mode with the given order of nested nulls.
    pub const fn with_nested_nulls(self, nested_nulls: NestedNulls) -> Self {
        Self {
            nested_nulls,
            ..self
        }
    }

    /// Returns how the mode treats signed zeros.
    pub fn signed_zeros(self) -> SignedZeros {
        self.signed_zeros
    }

    /// Returns how the mode treats `NaN` values.
    pub fn nan_order(self) -> NanOrder {
        self.nan_order
    }

    /// Returns how the mode orders nulls inside nested values.
    pub fn nested_nulls(self) -> NestedNulls {
        self.nested_nulls
    }

    /// Returns whether this mode is [`CompareMode::TOTAL_ORDER`].
    pub fn is_total_order(self) -> bool {
        self == Self::TOTAL_ORDER
    }

    /// Returns this mode with the default for each setting that cannot change a comparison of
    /// values of `dtype`.
    pub(crate) fn for_dtype(self, dtype: &DType) -> Self {
        let mut mode = self;
        if !contains_float(dtype) {
            mode.signed_zeros = SignedZeros::Distinct;
            mode.nan_order = NanOrder::TotalOrder;
        }
        if !is_nested(dtype) {
            mode.nested_nulls = NestedNulls::First;
        }
        mode
    }

    /// Returns whether comparing values of `dtype` with `operator` in this mode always gives the
    /// same result as in [`CompareMode::TOTAL_ORDER`].
    ///
    /// `constant` is the right-hand side when it is a known constant. Total-order kernels,
    /// statistics, and indexes can evaluate or prune a comparison that agrees.
    ///
    /// - With [`SignedZeros::Equal`], the constant must not be zero, so that both zeros are on the
    ///   same side of it.
    /// - With a [`NanOrder`] other than [`NanOrder::TotalOrder`], the operator must be `=` or `!=`
    ///   and the constant must not be `NaN`. A `NaN` on the left is then unequal in each mode.
    /// - With [`NestedNulls::Last`], the values must not be nested.
    pub fn agrees_with_total_order(
        self,
        operator: CompareOperator,
        dtype: &DType,
        constant: Option<&Scalar>,
    ) -> bool {
        let mode = self.for_dtype(dtype);
        if mode.is_total_order() {
            return true;
        }
        if mode.nested_nulls != NestedNulls::First {
            return false;
        }

        let Some(value) = constant.and_then(float_value) else {
            return false;
        };

        let zeros_agree = mode.signed_zeros == SignedZeros::Distinct || value != 0.0;
        let nans_agree = mode.nan_order == NanOrder::TotalOrder
            || (matches!(operator, CompareOperator::Eq | CompareOperator::NotEq)
                && !value.is_nan());

        zeros_agree && nans_agree
    }
}

impl Display for CompareMode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let zeros = match self.signed_zeros {
            SignedZeros::Distinct => "distinct",
            SignedZeros::Equal => "equal",
        };
        let nan = match self.nan_order {
            NanOrder::TotalOrder => "total_order",
            NanOrder::Canonical => "canonical",
            NanOrder::Ieee754 => "ieee754",
        };

        let nested_nulls = match self.nested_nulls {
            NestedNulls::First => "first",
            NestedNulls::Last => "last",
        };

        write!(f, "zeros={zeros}, nan={nan}, nested_nulls={nested_nulls}")
    }
}

/// Returns whether values of `dtype` are nested, so that they can hold nulls below the top level.
pub(crate) fn is_nested(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => is_nested(ext.storage_dtype()),
        DType::List(..) | DType::FixedSizeList(..) | DType::Struct(..) | DType::Map(..) => true,
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Union(..)
        | DType::Variant(_) => false,
    }
}

/// Returns whether values of `dtype` contain floats, so that their comparison depends on the
/// [`CompareMode`].
pub(crate) fn contains_float(dtype: &DType) -> bool {
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

impl From<CompareOperator> for Operator {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => Operator::Eq,
            CompareOperator::NotEq => Operator::NotEq,
            CompareOperator::Gt => Operator::Gt,
            CompareOperator::Gte => Operator::Gte,
            CompareOperator::Lt => Operator::Lt,
            CompareOperator::Lte => Operator::Lte,
        }
    }
}

impl TryFrom<Operator> for CompareOperator {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        value
            .to_compare_operator()
            .ok_or_else(|| vortex_error::vortex_err!("{value} is not a comparison operator"))
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
