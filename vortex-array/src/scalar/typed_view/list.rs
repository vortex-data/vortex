// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ListScalar`] typed view implementation.

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::builders::build_array_from_scalars;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing a list or fixed-size list (array) of elements.
///
/// We use the same [`ListScalar`] to represent both variants since a single list scalar's data is
/// identical to a single fixed-size list scalar.
///
/// This type provides a view into a list or fixed-size list scalar value which can contain zero or
/// more elements of the same type, or be null. If the `dtype` is a [`FixedSizeList`], then the
/// number of `elements` is equal to the `size` field of the [`FixedSizeList`].
///
/// The underlying data is always stored as a [`ScalarValue::Array`].
///
/// [`FixedSizeList`]: DType::FixedSizeList
#[derive(Debug, Clone)]
pub struct ListScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,

    /// A convenience field so that we do not have to unwrap and check the top-level `dtype` field
    /// every time we want to access this.
    element_dtype: &'a Arc<DType>,

    /// The underlying array, or `None` if the scalar is null.
    array: Option<&'a ArrayRef>,
}

impl<'a> ListScalar<'a> {
    /// Attempts to create a new [`ListScalar`] from a [`DType`] and optional [`ScalarValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not a [`DType::List`] or [`DType::FixedSizeList`].
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        let element_dtype = dtype
            .as_any_size_list_element_opt()
            .ok_or_else(|| vortex_err!("Expected list scalar, found {}", dtype))?;

        Ok(Self {
            dtype,
            element_dtype,
            array: value.map(|v| v.as_list()),
        })
    }

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
        self.array.map(|a| a.len()).unwrap_or(0)
    }

    /// Returns true if the list has no elements or is null.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.array.map(|a| a.is_empty()).unwrap_or(true)
    }

    /// Returns true if the list is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.array.is_none()
    }

    /// Returns the underlying [`ArrayRef`].
    ///
    /// This is useful for bulk operations (e.g., in builders) that can avoid expanding the array
    /// into individual scalars.
    pub fn as_array(&self) -> Option<&'a ArrayRef> {
        self.array
    }

    /// Returns the data type of the list's elements.
    pub fn element_dtype(&self) -> &DType {
        self.dtype
            .as_any_size_list_element_opt()
            .unwrap_or_else(|| vortex_panic!("`ListScalar` somehow had dtype {}", self.dtype))
            .as_ref()
    }

    /// Returns the element at the given index as a scalar.
    ///
    /// Returns `None` if the list is null or the index is out of bounds.
    pub fn element(&self, idx: usize) -> Option<Scalar> {
        let array = self.array?;
        if idx >= array.len() {
            return None;
        }

        Some(
            array
                .scalar_at(idx)
                .vortex_expect("index already bounds-checked"),
        )
    }

    /// Returns all elements in the list as a vector of scalars.
    ///
    /// Returns `None` if the list is null.
    ///
    /// This always materializes every element into a `Vec` by calling `Array::scalar_at()` for each
    /// element.
    ///
    /// Prefer [`as_array()`](Self::as_array) when the underlying array can be used directly (e.g.,
    /// in builders).
    pub fn elements(&self) -> Option<Vec<Scalar>> {
        let array = self.array?;
        Some(
            (0..array.len())
                .map(|i| {
                    array
                        .scalar_at(i)
                        .vortex_expect("index already bounds-checked")
                })
                .collect_vec(),
        )
    }

    /// Casts the list to the target [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the target [`DType`] is not a [`List`] or [`FixedSizeList`], or if trying to cast
    /// to a [`FixedSizeList`] with the incorrect number of elements.
    ///
    /// [`List`]: DType::List
    /// [`FixedSizeList`]: DType::FixedSizeList
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let target_element_dtype = dtype
            .as_any_size_list_element_opt()
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

        let array = self
            .array
            .ok_or_else(|| vortex_err!("nullness should be handled in Scalar::cast"))?;

        // Cast each element and build a new array with the target element dtype.
        let casted_elements = (0..array.len())
            .map(|i| {
                array
                    .scalar_at(i)
                    .vortex_expect("index already bounds-checked")
                    .cast(target_element_dtype)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let casted_array = build_array_from_scalars(target_element_dtype, &casted_elements);

        Scalar::try_new(dtype.clone(), Some(ScalarValue::Array(casted_array)))
    }
}

impl Display for ListScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.elements() {
            None => write!(f, "null"),
            Some(scalars) => {
                let type_str: &dyn Display = if let DType::FixedSizeList(_, size, _) = self.dtype {
                    &format!("fixed_size<{size}>")
                } else {
                    &""
                };

                write!(f, "{type_str}[{}]", scalars.iter().format(", "))
            }
        }
    }
}

impl PartialEq for ListScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return false;
        }

        match (self.array, other.array) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                // Compare element-by-element.
                a.len() == b.len()
                    && (0..a.len()).all(|i| a.scalar_at(i).ok() == b.scalar_at(i).ok())
            }
            _ => false,
        }
    }
}

impl Eq for ListScalar<'_> {}

/// Ord is not implemented since it's undefined for different element DTypes.
impl PartialOrd for ListScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self
            .element_dtype
            .eq_ignore_nullability(other.element_dtype)
        {
            return None;
        }

        match (self.array, other.array) {
            (None, None) => Some(Ordering::Equal),
            (None, Some(_)) => Some(Ordering::Less),
            (Some(_), None) => Some(Ordering::Greater),
            (Some(a), Some(b)) => {
                let min_len = a.len().min(b.len());

                // Compute the lexicographic ordering.
                for i in 0..min_len {
                    let a_scalar = a.scalar_at(i).vortex_expect(
                        "something happened with scalar_at in `PartialOrd` of `ListScalar`",
                    );
                    let b_scalar = b.scalar_at(i).vortex_expect(
                        "something happened with scalar_at in `PartialOrd` of `ListScalar`",
                    );

                    match a_scalar.partial_cmp(&b_scalar)? {
                        Ordering::Equal => continue,
                        ord => return Some(ord),
                    }
                }

                a.len().partial_cmp(&b.len())
            }
        }
    }
}

impl Hash for ListScalar<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.len().hash(state);

        if let Some(array) = self.array {
            for i in 0..array.len() {
                let scalar = array
                    .scalar_at(i)
                    .vortex_expect("something happened with `scalar_at` in `Hash` of `ListScalar`");
                scalar.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn test_list_scalar_creation() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
            Scalar::primitive(3i32, Nullability::NonNullable),
        ];
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();
        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());
        assert!(!list.is_null());
    }

    #[test]
    fn test_empty_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let list_scalar = Scalar::list_empty(element_dtype, Nullability::NonNullable);

        let list = list_scalar.as_list();
        assert_eq!(list.len(), 0);
        assert!(list.is_empty());
        assert!(!list.is_null());
    }

    #[test]
    fn test_null_list() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let null = Scalar::null(DType::List(element_dtype, Nullability::Nullable));

        let list = null.as_list();
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
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();

        // Test element access.
        let elem0 = list.element(0).unwrap();
        assert_eq!(elem0.as_primitive().typed_value::<i32>().unwrap(), 10);

        let elem1 = list.element(1).unwrap();
        assert_eq!(elem1.as_primitive().typed_value::<i32>().unwrap(), 20);

        let elem2 = list.element(2).unwrap();
        assert_eq!(elem2.as_primitive().typed_value::<i32>().unwrap(), 30);

        // Test out of bounds.
        assert!(list.element(3).is_none());
    }

    #[test]
    fn test_list_elements() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children = vec![
            Scalar::primitive(100i32, Nullability::NonNullable),
            Scalar::primitive(200i32, Nullability::NonNullable),
        ];
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();
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
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();
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
        let list_scalar1 =
            Scalar::list_from_scalars(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar2 =
            Scalar::list_from_scalars(element_dtype, children2, Nullability::NonNullable);

        let list1 = list_scalar1.as_list();
        let list2 = list_scalar2.as_list();

        assert_eq!(list1, list2);
    }

    #[test]
    fn test_list_inequality() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let children1 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(2i32, Nullability::NonNullable),
        ];
        let list_scalar1 =
            Scalar::list_from_scalars(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![
            Scalar::primitive(1i32, Nullability::NonNullable),
            Scalar::primitive(3i32, Nullability::NonNullable),
        ];
        let list_scalar2 =
            Scalar::list_from_scalars(element_dtype, children2, Nullability::NonNullable);

        let list1 = list_scalar1.as_list();
        let list2 = list_scalar2.as_list();

        assert_ne!(list1, list2);
    }

    #[test]
    fn test_list_partial_ord() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let children1 = vec![Scalar::primitive(1i32, Nullability::NonNullable)];
        let list_scalar1 =
            Scalar::list_from_scalars(element_dtype.clone(), children1, Nullability::NonNullable);

        let children2 = vec![Scalar::primitive(2i32, Nullability::NonNullable)];
        let list_scalar2 =
            Scalar::list_from_scalars(element_dtype, children2, Nullability::NonNullable);

        let list1 = list_scalar1.as_list();
        let list2 = list_scalar2.as_list();

        assert!(list1 < list2);
    }

    #[test]
    fn test_list_partial_ord_different_types() {
        let element_dtype1 = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let element_dtype2 = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));

        let children1 = vec![Scalar::primitive(1i32, Nullability::NonNullable)];
        let list_scalar1 =
            Scalar::list_from_scalars(element_dtype1, children1, Nullability::NonNullable);

        let children2 = vec![Scalar::primitive(1i64, Nullability::NonNullable)];
        let list_scalar2 =
            Scalar::list_from_scalars(element_dtype2, children2, Nullability::NonNullable);

        let list1 = list_scalar1.as_list();
        let list2 = list_scalar2.as_list();

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
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();

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
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

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
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();

        // Cast to list with i64 elements.
        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let casted = list.cast(&target_dtype).unwrap();
        let casted_list = casted.as_list();

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
        Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);
    }

    #[test]
    fn test_try_from_wrong_dtype() {
        let scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        assert!(scalar.as_list_opt().is_none());
    }

    #[test]
    fn test_string_list() {
        let element_dtype = Arc::new(DType::Utf8(Nullability::NonNullable));
        let children = vec![
            Scalar::utf8("hello".to_string(), Nullability::NonNullable),
            Scalar::utf8("world".to_string(), Nullability::NonNullable),
        ];
        let list_scalar =
            Scalar::list_from_scalars(element_dtype, children, Nullability::NonNullable);

        let list = list_scalar.as_list();
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

        let inner_list1 = Scalar::list_from_scalars(
            inner_element_dtype.clone(),
            vec![
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::primitive(2i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let inner_list2 = Scalar::list_from_scalars(
            inner_element_dtype,
            vec![
                Scalar::primitive(3i32, Nullability::NonNullable),
                Scalar::primitive(4i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );

        let outer_list = Scalar::list_from_scalars(
            inner_list_dtype,
            vec![inner_list1, inner_list2],
            Nullability::NonNullable,
        );

        let list = outer_list.as_list();
        assert_eq!(list.len(), 2);

        let nested_list = list.element(0).unwrap();
        let nested = nested_list.as_list();
        assert_eq!(nested.len(), 2);
    }
}
