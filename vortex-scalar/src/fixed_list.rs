// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;

use crate::Scalar;
use crate::ScalarValue;

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
#[derive(Debug, Clone)]
pub struct FixedSizeListScalar<'a> {
    pub(super) list_size: u32,
    pub(super) element_dtype: &'a Arc<DType>,
    pub(super) nullability: Nullability,
    pub(super) elements: Option<&'a [Option<ScalarValue>]>,
}

impl Display for FixedSizeListScalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.elements {
            None => write!(f, "null"),
            Some(elems) => {
                write!(
                    f,
                    "fixed_size<{}>[{}]",
                    self.list_size,
                    elems
                        .iter()
                        .map(|e| unsafe {
                            Scalar::new_unchecked(self.element_dtype().as_ref().clone(), e.clone())
                        })
                        .format(", ")
                )
            }
        }
    }
}

impl FixedSizeListScalar<'_> {
    /// Returns the number of elements in the list.
    pub fn list_size(&self) -> u32 {
        self.list_size
    }

    /// Returns the data type of the list elements.
    pub fn element_dtype(&self) -> &Arc<DType> {
        self.element_dtype
    }

    /// Returns the nullability of the list.
    pub fn nullability(&self) -> Nullability {
        self.nullability
    }

    /// Returns all elements as a slice of scalars, or `None` if null.
    pub fn elements(&self) -> Option<&[Option<ScalarValue>]> {
        self.elements
    }

    /// Returns the elements as an iterator of scalars, or `None` if null.
    pub fn elements_iter(&self) -> Option<impl Iterator<Item = Scalar>> {
        self.elements.as_ref().map(|elems| {
            elems
                .iter()
                .cloned()
                .map(|sv| unsafe { Scalar::new_unchecked(self.element_dtype.deref().clone(), sv) })
        })
    }
}

/// Helper functions to create a [`ListScalar`] as a [`Scalar`].
impl Scalar {
    pub fn fixed_size_list(
        element_dtype: Arc<DType>,
        children: Vec<Option<ScalarValue>>,
        nullability: Nullability,
    ) -> Self {
        let size = u32::try_from(children.len())
            .vortex_expect("tried to create a fixed-size list that was larger than u32");
        Self::try_new(
            DType::FixedSizeList(element_dtype, size, nullability),
            Some(ScalarValue::List(children)),
        )
        .vortex_expect("failed to create fixed-size list scalar")
    }
}

// TODO(v2): re-enable tests when removed API features are restored
/*
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

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
        use std::hash::Hash;
        use std::hash::Hasher;

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
        Scalar::list(element_dtype, children, Nullability::NonNullable);
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

*/
