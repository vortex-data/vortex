use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;

pub use scalar_type::ScalarType;
use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability};
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod arrow;
mod binary;
mod bool;
mod datafusion;
mod display;
mod extension;
mod list;
mod null;
mod primitive;
mod pvalue;
mod scalar_type;
mod scalarvalue;
#[cfg(feature = "serde")]
mod serde;
mod struct_;
mod utf8;

pub use binary::*;
pub use bool::*;
pub use extension::*;
pub use list::*;
pub use primitive::*;
pub use pvalue::*;
pub use scalarvalue::*;
pub use struct_::*;
pub use utf8::*;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

/// A single logical item, composed of both a [`ScalarValue`] and a logical [`DType`].
///
/// A [`ScalarValue`] is opaque, and should be accessed via one of the type-specific scalar wrappers
/// for example [`BoolScalar`], [`PrimitiveScalar`], etc.
///
/// Note that [`PartialOrd`] is implemented only for an exact match of the scalar's dtype,
/// including nullability. When the DType does match, ordering is nulls first (lowest), then the
/// natural ordering of the scalar value.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Scalar {
    dtype: DType,
    value: ScalarValue,
}

impl Scalar {
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        Self { dtype, value }
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }

    #[inline]
    pub fn into_parts(self) -> (DType, ScalarValue) {
        (self.dtype, self.value)
    }

    #[inline]
    pub fn into_value(self) -> ScalarValue {
        self.value
    }

    pub fn is_valid(&self) -> bool {
        !self.value.is_null()
    }

    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    pub fn null(dtype: DType) -> Self {
        assert!(dtype.is_nullable());
        Self {
            dtype,
            value: ScalarValue(InnerScalarValue::Null),
        }
    }

    pub fn null_typed<T: ScalarType>() -> Self {
        Self {
            dtype: T::dtype().as_nullable(),
            value: ScalarValue(InnerScalarValue::Null),
        }
    }

    pub fn cast(&self, target: &DType) -> VortexResult<Self> {
        if let DType::Extension(ext_dtype) = target {
            let storage_scalar = self.cast_to_non_extension(ext_dtype.storage_dtype())?;
            Ok(Scalar::extension(ext_dtype.clone(), storage_scalar))
        } else {
            self.cast_to_non_extension(target)
        }
    }

    fn cast_to_non_extension(&self, target: &DType) -> VortexResult<Self> {
        assert!(!matches!(target, DType::Extension(..)));
        if self.is_null() {
            if target.is_nullable() {
                return Ok(Scalar::new(target.clone(), self.value.clone()));
            } else {
                vortex_bail!("Can't cast null scalar to non-nullable type {}", target)
            }
        }

        if self.dtype().eq_ignore_nullability(target) {
            return Ok(Scalar::new(target.clone(), self.value.clone()));
        }

        match &self.dtype {
            DType::Null => unreachable!(), // handled by if is_null case
            DType::Bool(_) => self.as_bool().cast(target),
            DType::Primitive(..) => self.as_primitive().cast(target),
            DType::Utf8(_) => self.as_utf8().cast(target),
            DType::Binary(_) => self.as_binary().cast(target),
            DType::Struct(..) => self.as_struct().cast(target),
            DType::List(..) => self.as_list().cast(target),
            DType::Extension(..) => self.as_extension().cast(target),
        }
    }

    pub fn into_nullable(self) -> Self {
        Self {
            dtype: self.dtype.as_nullable(),
            value: self.value,
        }
    }
}

impl Scalar {
    pub fn as_bool(&self) -> BoolScalar {
        BoolScalar::try_from(self).vortex_expect("Failed to convert scalar to bool")
    }

    pub fn as_bool_opt(&self) -> Option<BoolScalar> {
        matches!(self.dtype, DType::Bool(..)).then(|| self.as_bool())
    }

    pub fn as_primitive(&self) -> PrimitiveScalar {
        PrimitiveScalar::try_from(self).vortex_expect("Failed to convert scalar to primitive")
    }

    pub fn as_primitive_opt(&self) -> Option<PrimitiveScalar> {
        matches!(self.dtype, DType::Primitive(..)).then(|| self.as_primitive())
    }

    pub fn as_utf8(&self) -> Utf8Scalar {
        Utf8Scalar::try_from(self).vortex_expect("Failed to convert scalar to utf8")
    }

    pub fn as_utf8_opt(&self) -> Option<Utf8Scalar> {
        matches!(self.dtype, DType::Utf8(..)).then(|| self.as_utf8())
    }

    pub fn as_binary(&self) -> BinaryScalar {
        BinaryScalar::try_from(self).vortex_expect("Failed to convert scalar to binary")
    }

    pub fn as_binary_opt(&self) -> Option<BinaryScalar> {
        matches!(self.dtype, DType::Binary(..)).then(|| self.as_binary())
    }

    pub fn as_struct(&self) -> StructScalar {
        StructScalar::try_from(self).vortex_expect("Failed to convert scalar to struct")
    }

    pub fn as_struct_opt(&self) -> Option<StructScalar> {
        matches!(self.dtype, DType::Struct(..)).then(|| self.as_struct())
    }

    pub fn as_list(&self) -> ListScalar {
        ListScalar::try_from(self).vortex_expect("Failed to convert scalar to list")
    }

    pub fn as_list_opt(&self) -> Option<ListScalar> {
        matches!(self.dtype, DType::List(..)).then(|| self.as_list())
    }

    pub fn as_extension(&self) -> ExtScalar {
        ExtScalar::try_from(self).vortex_expect("Failed to convert scalar to extension")
    }

    pub fn as_extension_opt(&self) -> Option<ExtScalar> {
        matches!(self.dtype, DType::Extension(..)).then(|| self.as_extension())
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        if self.dtype != other.dtype {
            return false;
        }

        match self.dtype() {
            DType::Null => true,
            DType::Bool(_) => self.as_bool() == other.as_bool(),
            DType::Primitive(..) => self.as_primitive() == other.as_primitive(),
            DType::Utf8(_) => self.as_utf8() == other.as_utf8(),
            DType::Binary(_) => self.as_binary() == other.as_binary(),
            DType::Struct(..) => self.as_struct() == other.as_struct(),
            DType::List(..) => self.as_list() == other.as_list(),
            DType::Extension(_) => self.as_extension() == other.as_extension(),
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dtype() != other.dtype() {
            return None;
        }

        match self.dtype() {
            DType::Null => Some(Ordering::Equal),
            DType::Bool(_) => self.as_bool().partial_cmp(&other.as_bool()),
            DType::Primitive(..) => self.as_primitive().partial_cmp(&other.as_primitive()),
            DType::Utf8(_) => self.as_utf8().partial_cmp(&other.as_utf8()),
            DType::Binary(_) => self.as_binary().partial_cmp(&other.as_binary()),
            DType::Struct(..) => self.as_struct().partial_cmp(&other.as_struct()),
            DType::List(..) => self.as_list().partial_cmp(&other.as_list()),
            DType::Extension(_) => self.as_extension().partial_cmp(&other.as_extension()),
        }
    }
}

impl Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.dtype() {
            DType::Null => self.dtype().hash(state), // Hash the dtype instead of the value
            DType::Bool(_) => self.as_bool().hash(state),
            DType::Primitive(..) => self.as_primitive().hash(state),
            DType::Utf8(_) => self.as_utf8().hash(state),
            DType::Binary(_) => self.as_binary().hash(state),
            DType::Struct(..) => self.as_struct().hash(state),
            DType::List(..) => self.as_list().hash(state),
            DType::Extension(_) => self.as_extension().hash(state),
        }
    }
}

impl AsRef<Self> for Scalar {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T> From<Option<T>> for Scalar
where
    T: ScalarType,
    Scalar: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value
            .map(Scalar::from)
            .map(|x| x.into_nullable())
            .unwrap_or_else(|| Scalar {
                dtype: T::dtype().as_nullable(),
                value: ScalarValue(InnerScalarValue::Null),
            })
    }
}

impl From<PrimitiveScalar<'_>> for Scalar {
    fn from(pscalar: PrimitiveScalar<'_>) -> Self {
        let dtype = pscalar.dtype().clone();
        let value = pscalar
            .pvalue()
            .map(|pvalue| ScalarValue(InnerScalarValue::Primitive(pvalue)))
            .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null));
        Self::new(dtype, value)
    }
}

macro_rules! from_vec_for_scalar {
    ($T:ty) => {
        impl From<Vec<$T>> for Scalar {
            fn from(value: Vec<$T>) -> Self {
                Scalar {
                    dtype: DType::List(Arc::from(<$T>::dtype()), Nullability::NonNullable),
                    value: ScalarValue(InnerScalarValue::List(
                        value
                            .into_iter()
                            .map(Scalar::from)
                            .map(|s| s.into_value())
                            .collect::<Arc<[_]>>(),
                    )),
                }
            }
        }
    };
}

// no From<Vec<u8>> because it could either be a List or a Buffer
from_vec_for_scalar!(u16);
from_vec_for_scalar!(u32);
from_vec_for_scalar!(u64);
from_vec_for_scalar!(usize); // For usize only, we implicitly cast for better ergonomics.
from_vec_for_scalar!(i8);
from_vec_for_scalar!(i16);
from_vec_for_scalar!(i32);
from_vec_for_scalar!(i64);
from_vec_for_scalar!(f16);
from_vec_for_scalar!(f32);
from_vec_for_scalar!(f64);
from_vec_for_scalar!(String);
from_vec_for_scalar!(BufferString);
from_vec_for_scalar!(ByteBuffer);

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_dtype::{DType, ExtDType, ExtID, Nullability, PType};

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    #[rstest]
    fn null_can_cast_to_anything_nullable(
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        source_dtype: DType,
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        target_dtype: DType,
    ) {
        assert_eq!(
            Scalar::null(source_dtype)
                .cast(&target_dtype)
                .unwrap()
                .dtype(),
            &target_dtype
        );
    }

    #[test]
    fn list_casts() {
        let list = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([ScalarValue(
                InnerScalarValue::Primitive(PValue::U16(6)),
            )]))),
        );

        let target_u32 = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u32).unwrap().dtype(), &target_u32);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert_eq!(
            list.cast(&target_u32_nonnull).unwrap().dtype(),
            &target_u32_nonnull
        );

        let target_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::NonNullable,
        );
        assert_eq!(list.cast(&target_nonnull).unwrap().dtype(), &target_nonnull);

        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u8).unwrap().dtype(), &target_u8);

        let list_with_null = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([
                ScalarValue(InnerScalarValue::Primitive(PValue::U16(6))),
                ScalarValue(InnerScalarValue::Null),
            ]))),
        );
        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list_with_null.cast(&target_u8).unwrap().dtype(), &target_u8);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert!(list_with_null.cast(&target_u32_nonnull).is_err());
    }

    #[test]
    fn cast_to_from_extension_types() {
        let apples = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            None,
        );
        let ext_dtype = DType::Extension(Arc::from(apples.clone()));
        let ext_scalar = Scalar::new(ext_dtype.clone(), ScalarValue(InnerScalarValue::Bool(true)));
        let storage_scalar = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(1000))),
        );

        // to self
        let expected_dtype = &ext_dtype;
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // to nullable self
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type
        let expected_dtype = apples.storage_dtype();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type, nullable
        let expected_dtype = &apples.storage_dtype().as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension, nullable
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *compatible* storage type to extension
        let storage_scalar_u64 = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(1000))),
        );
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar_u64.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *incompatible* storage type to extension
        let apples_u8 = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable)),
            None,
        );
        let expected_dtype = &DType::Extension(Arc::from(apples_u8));
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result.as_ref().is_err_and(|err| {
                err
                    .to_string()
                    .contains("Can't cast u16 scalar 1000_u16 to u8 (cause: Cannot read primitive value U16(1000) as u8")
            }),
            "{:?}",
            result
        );
    }
}
