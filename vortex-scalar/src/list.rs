// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex_dtype::{DType, Nullability};
use vortex_error::{
    VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_err, vortex_panic,
};

use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing a list or fixed-size list (array) of elements.
///
/// We use the same [`ListScalar`] to represent both variants since a single list scalar's data is
/// identical to a single fixed-size list scalar.
///
/// This type provides a view into a list or fixed-size list scalar value which can contain zero or
/// more elements of the same type, or be null. If the `dtype` is a [`FixedSizeList`], then the
/// number of `elements` is equal to the `size` field of the [`FixedSizeList`].
///
/// [`FixedSizeList`]: DType::FixedSizeList
#[derive(Debug)]
pub struct ListScalar<'a> {
    dtype: &'a DType,
    element_dtype: &'a Arc<DType>,
    elements: Option<Arc<[ScalarValue]>>,
}

impl Display for ListScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.elements {
            None => write!(f, "null"),
            Some(elems) => {
                let fixed_size_list_str: &dyn Display =
                    if let DType::FixedSizeList(_, size, _) = self.dtype {
                        &format!("fixed_size<{size}>")
                    } else {
                        &""
                    };

                write!(
                    f,
                    "{fixed_size_list_str}[{}]",
                    elems
                        .iter()
                        .map(|e| Scalar::new(self.element_dtype().clone(), e.clone()))
                        .format(", ")
                )
            }
        }
    }
}

impl PartialEq for ListScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.elements() == other.elements()
    }
}

impl Eq for ListScalar<'_> {}

/// Ord is not implemented since it's undefined for different element DTypes
impl PartialOrd for ListScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self
            .element_dtype
            .eq_ignore_nullability(other.element_dtype)
        {
            return None;
        }
        self.elements().partial_cmp(&other.elements())
    }
}

impl Hash for ListScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.elements().hash(state);
    }
}

impl<'a> ListScalar<'a> {
    /// Returns the data type of this list scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the number of elements in the list.
    ///
    /// Returns 0 if the list is null.
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Returns true if the list has no elements or is null.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self.elements.as_ref() {
            None => true,
            Some(l) => l.is_empty(),
        }
    }

    /// Returns true if the list is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.elements.is_none()
    }

    /// Returns the data type of the list's elements.
    pub fn element_dtype(&self) -> &DType {
        self.dtype
            .as_list_element_opt()
            .unwrap_or_else(|| vortex_panic!("`ListScalar` somehow had dtype {}", self.dtype))
            .as_ref()
    }

    /// Returns the element at the given index as a scalar.
    ///
    /// Returns None if the list is null or the index is out of bounds.
    pub fn element(&self, idx: usize) -> Option<Scalar> {
        self.elements
            .as_ref()
            .and_then(|l| l.get(idx))
            .map(|value| Scalar::new(self.element_dtype().clone(), value.clone()))
    }

    /// Returns all elements in the list as a vector of scalars.
    ///
    /// Returns None if the list is null.
    pub fn elements(&self) -> Option<Vec<Scalar>> {
        self.elements.as_ref().map(|elems| {
            elems
                .iter()
                .map(|e| Scalar::new(self.element_dtype().clone(), e.clone()))
                .collect_vec()
        })
    }

    /// Casts the list to the target [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the target [`DType`] is not a [`List`]: or [`FixedSizeList`], or if trying to cast
    /// to a [`FixedSizeList`] with the incorrect number of elements.
    ///
    /// [`List`]: DType::List
    /// [`FixedSizeList`]: DType::FixedSizeList
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let target_element_dtype = dtype
            .as_list_element_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "Cannot cast {} to {}: list can only be cast to a list or fixed-size list",
                    self.dtype(),
                    dtype
                )
            })?
            .as_ref();

        if let DType::FixedSizeList(_, size, _) = dtype
            && *size as usize != self.len()
        {
            vortex_bail!(
                "tried to cast to a `FixedSizeList[{size}]` but had {} elements",
                self.len()
            )
        }

        Ok(Scalar::new(
            dtype.clone(),
            ScalarValue(InnerScalarValue::List(
                self.elements
                    .as_ref()
                    .vortex_expect("nullness handled in Scalar::cast")
                    .iter()
                    .map(|element| {
                        // Recursively cast the elements of the list.
                        Scalar::new(DType::clone(self.element_dtype), element.clone())
                            .cast(target_element_dtype)
                            .map(|x| x.value().clone())
                    })
                    .collect::<VortexResult<Arc<[ScalarValue]>>>()?,
            )),
        ))
    }
}

/// A helper enum for creating a [`ListScalar`].
enum ListKind {
    Variable,
    FixedSize,
}

/// Helper functions to create a [`ListScalar`] as a [`Scalar`].
impl Scalar {
    fn create_list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
        list_kind: ListKind,
    ) -> Self {
        let element_dtype = element_dtype.into();

        let children: Arc<[ScalarValue]> = children
            .into_iter()
            .map(|child| {
                if child.dtype() != &*element_dtype {
                    vortex_panic!(
                        "tried to create list of {} with values of type {}",
                        element_dtype,
                        child.dtype()
                    );
                }
                child.into_value()
            })
            .collect();
        let size: u32 = children
            .len()
            .try_into()
            .vortex_expect("tried to create a list that was too large");

        let dtype = match list_kind {
            ListKind::Variable => DType::List(element_dtype, nullability),
            ListKind::FixedSize => DType::FixedSizeList(element_dtype, size, nullability),
        };

        Self::new(dtype, ScalarValue(InnerScalarValue::List(children)))
    }

    /// Creates a new list scalar with the given element type and children.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list(element_dtype, children, nullability, ListKind::Variable)
    }

    /// Creates a new empty list scalar with the given element type.
    pub fn list_empty(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::create_list(element_dtype, vec![], nullability, ListKind::Variable)
    }

    /// Creates a new fixed-size list scalar with the given element type and children.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn fixed_size_list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list(element_dtype, children, nullability, ListKind::FixedSize)
    }
}

impl<'a> TryFrom<&'a Scalar> for ListScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        let element_dtype = value
            .dtype()
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("Expected list scalar, found {}", value.dtype()))?;

        Ok(Self {
            dtype: value.dtype(),
            element_dtype,
            elements: value.value().as_list()?.cloned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType};

    use super::*;

    #[test]
    fn test_list_scalar_creation() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
            Scalar::primitive(3i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();
        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());
        assert!(!list.is_null());
    }

    #[test]
    fn test_empty_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let list_scalar = Scalar::list_empty(element_dtype, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(!list.is_null());
    }

    #[test]
    fn test_null_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let null = Scalar::null(DType::List(element_dtype, Nullability::Nullable));

        let list = ListScalar::try_from(&null).unwrap();
        assert!(list.is_empty());
        assert!(list.is_null());
    }

    #[test]
    fn test_list_element_access() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(10i32, Nullability::NonNullable),
            Scalar::primitive(20i32, Nullability::NonNullable),
            Scalar::primitive(30i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();

        // Test element access
        let elem0 = list.element(0).unwrap();
        assert_eq!(elem0.as_primitive().typed_value::<i32>().unwrap(), 10);

        let elem1 = list.element(1).unwrap();
        assert_eq!(elem1.as_primitive().typed_value::<i32>().unwrap(), 20);

        let elem2 = list.element(2).unwrap();
        assert_eq!(elem2.as_primitive().typed_value::<i32>().unwrap(), 30);

        // Test out of bounds
        assert!(list.element(3).is_none());
    }

    #[test]
    fn test_list_elements() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(100i32, Nullability::NonNullable),
            Scalar::primitive(200i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();
        let elements = list.elements().unwrap();

        assert_eq!(elements.len(), 2);
        assert_eq!(
            elements[0].as_primitive().typed_value::<i32>().unwrap(),
            100
        );
        assert_eq!(
            elements[1].as_primitive().typed_value::<i32>().unwrap(),
            200
        );
    }

    #[test]
    fn test_list_display() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();
        let display = format!("{list}");
        assert!(display.contains("1"));
        assert!(display.contains("2"));
    }

    #[test]
    fn test_list_equality() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children1 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar1 = Scalar::list(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar2 = Scalar::list(element_dtype, children2, Nullability::NonNullable);

        let list1 = ListScalar::try_from(&list_scalar1).unwrap();
        let list2 = ListScalar::try_from(&list_scalar2).unwrap();

        assert_eq!(list1, list2);
    }

    #[test]
    fn test_list_inequality() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children1 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar1 = Scalar::list(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(3i32, Nullability::NonNullable),
        ];
        let list_scalar2 = Scalar::list(element_dtype, children2, Nullability::NonNullable);

        let list1 = ListScalar::try_from(&list_scalar1).unwrap();
        let list2 = ListScalar::try_from(&list_scalar2).unwrap();

        assert_ne!(list1, list2);
    }

    #[test]
    fn test_list_partial_ord() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let children1 = vec![Scalar::primitive(1i32, Nullability::NonNullable)];
        let list_scalar1 = Scalar::list(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![Scalar::primitive(2i32, Nullability::NonNullable)];
        let list_scalar2 = Scalar::list(element_dtype, children2, Nullability::NonNullable);

        let list1 = ListScalar::try_from(&list_scalar1).unwrap();
        let list2 = ListScalar::try_from(&list_scalar2).unwrap();

        assert!(list1 < list2);
    }

    #[test]
    fn test_list_partial_ord_different_types() {
        let element_dtype1 = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let element_dtype2 = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));

        let children1 = vec![Scalar::primitive(1i32, Nullability::NonNullable)];
        let list_scalar1 = Scalar::list(element_dtype1, children1, Nullability::NonNullable);

        let children2 = vec![Scalar::primitive(1i64, Nullability::NonNullable)];
        let list_scalar2 = Scalar::list(element_dtype2, children2, Nullability::NonNullable);

        let list1 = ListScalar::try_from(&list_scalar1).unwrap();
        let list2 = ListScalar::try_from(&list_scalar2).unwrap();

        assert!(list1.partial_cmp(&list2).is_none());
    }

    #[test]
    fn test_list_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();

        let mut hasher1 = DefaultHasher::new();
        list.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        list.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_vec_conversion() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(10i32, Nullability::NonNullable),
            Scalar::primitive(20i32, Nullability::NonNullable),
            Scalar::primitive(30i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let vec: Vec<i32> = Vec::try_from(&list_scalar).unwrap();
        assert_eq!(vec, vec![10, 20, 30]);
    }

    #[test]
    fn test_vec_conversion_empty_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let list_scalar = Scalar::list_empty(element_dtype, Nullability::Nullable);

        let result: VortexResult<Vec<i32>> = Vec::try_from(&list_scalar);
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_list_cast() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();

        // Cast to list with i64 elements
        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let casted = list.cast(&target_dtype).unwrap();
        let casted_list = ListScalar::try_from(&casted).unwrap();

        assert_eq!(casted_list.len(), 2);
        let elem0 = casted_list.element(0).unwrap();
        assert_eq!(elem0.as_primitive().typed_value::<i64>().unwrap(), 1);
    }

    #[test]
    #[should_panic(expected = "tried to create list of i32 with values of type i64")]
    fn test_list_wrong_element_type_panic() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i64, Nullability::NonNullable), // Wrong type!
        ];
        let _ = Scalar::list(element_dtype, children, Nullability::NonNullable);
    }

    #[test]
    fn test_try_from_wrong_dtype() {
        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let result = ListScalar::try_from(&scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_string_list() {
        let element_dtype = Arc::new(DType::Utf8(Nullability::NonNullable));
        let children = vec![
            Scalar::utf8("hello".to_string(), Nullability::NonNullable),
            Scalar::utf8("world".to_string(), Nullability::NonNullable),
        ];
        let list_scalar = Scalar::list(element_dtype, children, Nullability::NonNullable);

        let list = ListScalar::try_from(&list_scalar).unwrap();
        assert_eq!(list.len(), 2);

        let elem0 = list.element(0).unwrap();
        assert_eq!(elem0.as_utf8().value().unwrap().as_str(), "hello");

        let elem1 = list.element(1).unwrap();
        assert_eq!(elem1.as_utf8().value().unwrap().as_str(), "world");
    }

    #[test]
    fn test_nested_lists() {
        let inner_element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let inner_list_dtype = Arc::new(DType::List(
            inner_element_dtype.clone(),
            Nullability::NonNullable,
        ));

        let inner_list1 = Scalar::list(
            inner_element_dtype.clone(),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::list(
            inner_element_dtype,
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_list = Scalar::list(
            inner_list_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        let list = ListScalar::try_from(&outer_list).unwrap();
        assert_eq!(list.len(), 2);

        let nested_list = list.element(0).unwrap();
        let nested = ListScalar::try_from(&nested_list).unwrap();
        assert_eq!(nested.len(), 2);
    }
}
