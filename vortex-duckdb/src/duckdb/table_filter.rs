// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ptr;

use cpp::duckdb_vx_table_filter;
use num_traits::AsPrimitive;
use vortex::error::vortex_panic;

use crate::cpp;
use crate::cpp::idx_t;
use crate::duckdb::Expression;
use crate::duckdb::ExpressionRef;
use crate::duckdb::Value;
use crate::duckdb::ValueRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(TableFilterSet, cpp::duckdb_vx_table_filter_set, |_| {});

impl TableFilterSetRef {
    pub fn get(&self, index: u64) -> Option<(idx_t, &TableFilterRef)> {
        let mut filter_set: duckdb_vx_table_filter = ptr::null_mut();

        let column_index = unsafe {
            cpp::duckdb_vx_table_filter_set_get(self.as_ptr(), index.as_(), &raw mut filter_set)
        };

        if filter_set.is_null() {
            None
        } else {
            Some(unsafe { (column_index, TableFilter::borrow(filter_set)) })
        }
    }

    pub fn len(&self) -> idx_t {
        unsafe { cpp::duckdb_vx_table_filter_set_size(self.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> IntoIterator for &'a TableFilterSetRef {
    type Item = (idx_t, &'a TableFilterRef);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new((0..self.len()).map(move |i| {
            self.get(i)
                .unwrap_or_else(|| vortex_panic!("inside filter set bounds {i}"))
        }))
    }
}

impl Debug for TableFilterSetRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self).finish()
    }
}

lifetime_wrapper!(TableFilter, duckdb_vx_table_filter, |_| {});

impl TableFilterRef {
    pub fn as_class(&self) -> TableFilterClass<'_> {
        match unsafe { cpp::duckdb_vx_table_filter_get_type(self.as_ptr()) } {
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_CONSTANT_COMPARISON => {
                let mut out = cpp::duckdb_vx_table_filter_constant {
                    value: ptr::null_mut(),
                    comparison_type: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                };
                unsafe { cpp::duckdb_vx_table_filter_get_constant(self.as_ptr(), &raw mut out) };

                TableFilterClass::ConstantComparison(ConstantComparison {
                    value: unsafe { Value::borrow(out.value) },
                    operator: out.comparison_type,
                })
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IS_NULL => {
                TableFilterClass::IsNull
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IS_NOT_NULL => {
                TableFilterClass::IsNotNull
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_CONJUNCTION_OR => {
                let mut out = cpp::duckdb_vx_table_filter_conjunction {
                    children: ptr::null_mut(),
                    children_count: 0,
                };
                unsafe {
                    cpp::duckdb_vx_table_filter_get_conjunction_or(self.as_ptr(), &raw mut out)
                };

                TableFilterClass::ConjunctionOr(Conjunction {
                    children: if out.children_count == 0 {
                        &[]
                    } else {
                        // SAFETY: `children_count > 0`, so DuckDB populated `out.children` with a
                        // valid pointer to `out.children_count` child handles owned by `self`.
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) }
                    },
                })
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_CONJUNCTION_AND => {
                let mut out = cpp::duckdb_vx_table_filter_conjunction {
                    children: ptr::null_mut(),
                    children_count: 0,
                };
                unsafe {
                    cpp::duckdb_vx_table_filter_get_conjunction_and(self.as_ptr(), &raw mut out)
                };

                TableFilterClass::ConjunctionAnd(Conjunction {
                    children: if out.children_count == 0 {
                        &[]
                    } else {
                        // SAFETY: `children_count > 0`, so DuckDB populated `out.children` with a
                        // valid pointer to `out.children_count` child handles owned by `self`.
                        unsafe { std::slice::from_raw_parts(out.children, out.children_count) }
                    },
                })
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_STRUCT_EXTRACT => {
                let mut out = cpp::duckdb_vx_table_filter_struct_extract {
                    child_filter: ptr::null_mut(),
                    child_name: ptr::null_mut(),
                    child_name_len: 0,
                };
                unsafe {
                    cpp::duckdb_vx_table_filter_get_struct_extract(self.as_ptr(), &raw mut out)
                };

                let name = unsafe {
                    str::from_utf8_unchecked(std::slice::from_raw_parts(
                        out.child_name.cast::<u8>(),
                        out.child_name_len,
                    ))
                };
                let child_filter = unsafe { TableFilter::borrow(out.child_filter) };

                TableFilterClass::StructExtract(name, child_filter)
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_OPTIONAL_FILTER => {
                let child_filter = unsafe {
                    TableFilter::borrow(cpp::duckdb_vx_table_filter_get_optional(self.as_ptr()))
                };
                TableFilterClass::Optional(child_filter)
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IN_FILTER => {
                let mut out = cpp::duckdb_vx_table_filter_in_filter {
                    values: ptr::null_mut(),
                    values_count: 0,
                };
                unsafe { cpp::duckdb_vx_table_filter_get_in_filter(self.as_ptr(), &raw mut out) };

                TableFilterClass::InFilter(Values {
                    values: out.values,
                    values_count: out.values_count,
                    _phantom: Default::default(),
                })
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_DYNAMIC_FILTER => {
                let mut out = cpp::duckdb_vx_table_filter_dynamic {
                    data: ptr::null_mut(),
                    comparison_type: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                };
                unsafe { cpp::duckdb_vx_table_filter_get_dynamic(self.as_ptr(), &raw mut out) };
                TableFilterClass::Dynamic(DynamicFilter {
                    data: unsafe { DynamicFilterData::own(out.data) },
                    operator: out.comparison_type,
                })
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_EXPRESSION_FILTER => {
                let expr = unsafe {
                    Expression::borrow(cpp::duckdb_vx_table_filter_get_expression(self.as_ptr()))
                };
                TableFilterClass::ExpressionRef(expr)
            }
            cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_BLOOM_FILTER => {
                // TODO(aduffy): actually extract these parameters
                TableFilterClass::Bloom
            }
        }
    }
}

impl Debug for TableFilterRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_vx_table_filter_to_debug_string(self.as_ptr()) };
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let result = write!(f, "{}", cstr.to_string_lossy());
        unsafe { cpp::duckdb_free(ptr.cast_mut().cast()) };
        result
    }
}

pub enum TableFilterClass<'a> {
    ConstantComparison(ConstantComparison<'a>),
    IsNull,
    IsNotNull,
    ConjunctionOr(Conjunction<'a>),
    ConjunctionAnd(Conjunction<'a>),
    StructExtract(&'a str, &'a TableFilterRef),
    Optional(&'a TableFilterRef),
    InFilter(Values<'a>),
    Dynamic(DynamicFilter),
    ExpressionRef(&'a ExpressionRef),
    Bloom,
}

pub struct ConstantComparison<'a> {
    pub value: &'a ValueRef,
    pub operator: cpp::DUCKDB_VX_EXPR_TYPE,
}

pub struct Conjunction<'a> {
    children: &'a [duckdb_vx_table_filter],
}

impl<'a> Conjunction<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a TableFilterRef> + 'a {
        self.children
            .iter()
            .map(|&child| unsafe { TableFilter::borrow(child) })
    }
}

pub struct Values<'a> {
    // We cannot use a Rust slice since the values are not individually heap-allocated.
    values: cpp::duckdb_vx_values_vec,
    values_count: usize,
    _phantom: PhantomData<&'a ()>,
}

struct ValuesIterator<'a> {
    values: cpp::duckdb_vx_values_vec,
    values_count: usize,
    index: usize,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Iterator for ValuesIterator<'a> {
    type Item = &'a ValueRef;

    fn next(&mut self) -> Option<Self::Item> {
        (self.index < self.values_count).then(|| {
            let value = unsafe {
                Value::borrow(cpp::duckdb_vx_values_vec_get(self.values, self.index as _))
            };
            self.index += 1;
            value
        })
    }
}

impl<'a> Values<'a> {
    pub fn iter(&self) -> impl Iterator<Item = &'a ValueRef> {
        ValuesIterator {
            values: self.values,
            values_count: self.values_count,
            index: 0,
            _phantom: Default::default(),
        }
    }
}

pub struct DynamicFilter {
    pub data: DynamicFilterData,
    pub operator: cpp::DUCKDB_VX_EXPR_TYPE,
}

lifetime_wrapper!(
    /// A handle to mutable dynamic filter data.
    DynamicFilterData,
    cpp::duckdb_vx_dynamic_filter_data,
    cpp::duckdb_vx_dynamic_filter_data_free
);

/// This handle wraps a single shared_ptr on C++ side, so we can assert is Send + Sync
unsafe impl Send for DynamicFilterData {}
unsafe impl Sync for DynamicFilterData {}

impl DynamicFilterDataRef {
    /// Fetches the latest value from the dynamic filter data, if it has been initialized.
    pub fn latest(&self) -> Option<Value> {
        let ptr = unsafe { cpp::duckdb_vx_dynamic_filter_data_get_value(self.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { Value::own(ptr) })
    }
}

/// cbindgen:ignore
#[cfg(all(test, table_filter_test_support))]
pub(crate) mod test_support {
    use std::ffi::CString;
    use std::ffi::c_char;
    use std::ffi::c_void;
    use std::ptr;

    use vortex::error::VortexExpect;

    use crate::cpp;
    use crate::cpp::DUCKDB_VX_EXPR_TYPE;
    use crate::cpp::idx_t;
    use crate::duckdb::LogicalTypeRef;
    use crate::duckdb::TableFilter;
    use crate::duckdb::TableFilterRef;
    use crate::duckdb::Value;

    type TestDynamicFilterDataPtr = *mut c_void;

    unsafe extern "C-unwind" {
        fn duckdb_vx_test_table_filter_destroy(filter: *mut cpp::duckdb_vx_table_filter);
        fn duckdb_vx_test_dynamic_filter_data_destroy(data: *mut TestDynamicFilterDataPtr);
        fn duckdb_vx_test_dynamic_filter_data_set_value(
            data: TestDynamicFilterDataPtr,
            value: cpp::duckdb_value,
        );
        fn duckdb_vx_test_table_filter_create_constant(
            comparison_type: DUCKDB_VX_EXPR_TYPE,
            value: cpp::duckdb_value,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_is_null() -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_is_not_null() -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_conjunction_and(
            children: *mut cpp::duckdb_vx_table_filter,
            count: usize,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_conjunction_or(
            children: *mut cpp::duckdb_vx_table_filter,
            count: usize,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_struct_extract(
            child_idx: idx_t,
            child_name: *const c_char,
            child_filter: cpp::duckdb_vx_table_filter,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_optional(
            child_filter: cpp::duckdb_vx_table_filter,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_in(
            values: *mut cpp::duckdb_value,
            count: usize,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_dynamic(
            comparison_type: DUCKDB_VX_EXPR_TYPE,
            initial_value: cpp::duckdb_value,
            data_out: *mut TestDynamicFilterDataPtr,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_bloom() -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_expr_comparison(
            comparison_type: DUCKDB_VX_EXPR_TYPE,
            value: cpp::duckdb_value,
            wrap_optional: bool,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_expr_in(
            col_type: cpp::duckdb_logical_type,
            values: *mut cpp::duckdb_value,
            count: usize,
            wrap_optional: bool,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_expr_dynamic(
            comparison_type: DUCKDB_VX_EXPR_TYPE,
            initial_value: cpp::duckdb_value,
            wrap_optional: bool,
            data_out: *mut TestDynamicFilterDataPtr,
        ) -> cpp::duckdb_vx_table_filter;
        fn duckdb_vx_test_table_filter_create_expr_unsupported() -> cpp::duckdb_vx_table_filter;
    }

    /// Owned handle to a test-created DuckDB `DynamicFilterData`, allowing tests to push values
    /// into a live dynamic filter.
    pub(crate) struct TestDynamicFilterData(TestDynamicFilterDataPtr);

    impl TestDynamicFilterData {
        pub(crate) fn set_value(&self, value: &Value) {
            // SAFETY: `self.0` is an owned handle allocated by the C++ test helper and `value`
            // is a valid DuckDB value pointer cloned by `DynamicFilterData::SetValue`.
            unsafe {
                duckdb_vx_test_dynamic_filter_data_set_value(self.0, value.as_ptr());
            }
        }
    }

    impl Drop for TestDynamicFilterData {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` was allocated by `duckdb_vx_test_table_filter_create_*dynamic`
                // and has not been freed yet; destroy nulls it out.
                unsafe { duckdb_vx_test_dynamic_filter_data_destroy(&raw mut self.0) };
            }
        }
    }

    /// Owned DuckDB `TableFilter` handle for unit tests (frees on drop unless transferred).
    pub(crate) struct OwnedTableFilter(cpp::duckdb_vx_table_filter);

    impl OwnedTableFilter {
        pub(crate) fn as_ref(&self) -> &TableFilterRef {
            // SAFETY: `self.0` is a live owned `TableFilter` pointer valid for `&self`.
            unsafe { TableFilter::borrow(self.0) }
        }

        fn into_raw(mut self) -> cpp::duckdb_vx_table_filter {
            let ptr = self.0;
            self.0 = ptr::null_mut();
            ptr
        }

        pub(crate) fn constant(op: DUCKDB_VX_EXPR_TYPE, value: &Value) -> Self {
            // SAFETY: `value.as_ptr()` is a valid DuckDB value; C++ copies it and returns an owned filter.
            Self(unsafe { duckdb_vx_test_table_filter_create_constant(op, value.as_ptr()) })
        }

        pub(crate) fn is_null() -> Self {
            // SAFETY: Returns a freshly allocated owned `TableFilter`.
            Self(unsafe { duckdb_vx_test_table_filter_create_is_null() })
        }

        pub(crate) fn is_not_null() -> Self {
            // SAFETY: Returns a freshly allocated owned `TableFilter`.
            Self(unsafe { duckdb_vx_test_table_filter_create_is_not_null() })
        }

        pub(crate) fn conjunction_and(children: Vec<Self>) -> Self {
            let mut raw: Vec<_> = children.into_iter().map(Self::into_raw).collect();
            // SAFETY: `raw` holds `raw.len()` valid owned child pointers whose ownership transfers
            // to the returned conjunction filter.
            Self(unsafe {
                duckdb_vx_test_table_filter_create_conjunction_and(raw.as_mut_ptr(), raw.len())
            })
        }

        pub(crate) fn conjunction_or(children: Vec<Self>) -> Self {
            let mut raw: Vec<_> = children.into_iter().map(Self::into_raw).collect();
            // SAFETY: `raw` holds `raw.len()` valid owned child pointers whose ownership transfers
            // to the returned conjunction filter.
            Self(unsafe {
                duckdb_vx_test_table_filter_create_conjunction_or(raw.as_mut_ptr(), raw.len())
            })
        }

        pub(crate) fn struct_extract(child_idx: idx_t, child_name: &str, child: Self) -> Self {
            let name = CString::new(child_name)
                .ok()
                .vortex_expect("valid C string");
            // SAFETY: `name` is NUL-terminated for the call and `child.into_raw()` transfers
            // ownership of the child filter to the returned struct filter.
            Self(unsafe {
                duckdb_vx_test_table_filter_create_struct_extract(
                    child_idx,
                    name.as_ptr(),
                    child.into_raw(),
                )
            })
        }

        pub(crate) fn optional(child: Self) -> Self {
            // SAFETY: `child.into_raw()` transfers ownership of the child filter to the returned optional filter.
            Self(unsafe { duckdb_vx_test_table_filter_create_optional(child.into_raw()) })
        }

        pub(crate) fn in_filter(values: &[Value]) -> Self {
            let mut ptrs: Vec<_> = values.iter().map(|v| v.as_ptr()).collect();
            // SAFETY: `ptrs` holds `ptrs.len()` valid DuckDB value pointers cloned by the C++ helper.
            Self(unsafe { duckdb_vx_test_table_filter_create_in(ptrs.as_mut_ptr(), ptrs.len()) })
        }

        pub(crate) fn dynamic(
            op: DUCKDB_VX_EXPR_TYPE,
            initial: &Value,
        ) -> (Self, TestDynamicFilterData) {
            let mut data: TestDynamicFilterDataPtr = ptr::null_mut();
            // SAFETY: `initial.as_ptr()` is valid and `&raw mut data` receives an owned handle.
            let filter = Self(unsafe {
                duckdb_vx_test_table_filter_create_dynamic(op, initial.as_ptr(), &raw mut data)
            });
            (filter, TestDynamicFilterData(data))
        }

        pub(crate) fn bloom() -> Self {
            // SAFETY: Returns a freshly allocated owned `TableFilter`.
            Self(unsafe { duckdb_vx_test_table_filter_create_bloom() })
        }

        pub(crate) fn expr_comparison(
            op: DUCKDB_VX_EXPR_TYPE,
            value: &Value,
            wrap_optional: bool,
        ) -> Self {
            // SAFETY: `value.as_ptr()` is a valid DuckDB value cloned by the C++ helper.
            Self(unsafe {
                duckdb_vx_test_table_filter_create_expr_comparison(
                    op,
                    value.as_ptr(),
                    wrap_optional,
                )
            })
        }

        pub(crate) fn expr_in(
            col_type: &LogicalTypeRef,
            values: &[Value],
            wrap_optional: bool,
        ) -> Self {
            let mut ptrs: Vec<_> = values.iter().map(|v| v.as_ptr()).collect();
            // SAFETY: `col_type` and `ptrs[..ptrs.len()]` are valid DuckDB handles borrowed for the call.
            Self(unsafe {
                duckdb_vx_test_table_filter_create_expr_in(
                    col_type.as_ptr(),
                    ptrs.as_mut_ptr(),
                    ptrs.len(),
                    wrap_optional,
                )
            })
        }

        pub(crate) fn expr_dynamic(
            op: DUCKDB_VX_EXPR_TYPE,
            initial: &Value,
            wrap_optional: bool,
        ) -> (Self, TestDynamicFilterData) {
            let mut data: TestDynamicFilterDataPtr = ptr::null_mut();
            // SAFETY: `initial.as_ptr()` is valid and `&raw mut data` receives an owned handle.
            let filter = Self(unsafe {
                duckdb_vx_test_table_filter_create_expr_dynamic(
                    op,
                    initial.as_ptr(),
                    wrap_optional,
                    &raw mut data,
                )
            });
            (filter, TestDynamicFilterData(data))
        }

        pub(crate) fn expr_unsupported() -> Self {
            // SAFETY: Returns a freshly allocated owned `TableFilter`.
            Self(unsafe { duckdb_vx_test_table_filter_create_expr_unsupported() })
        }
    }

    impl Drop for OwnedTableFilter {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` is still owned by `self` (not transferred via `into_raw`) and
                // was allocated by a C++ test helper; destroy frees and nulls it.
                unsafe { duckdb_vx_test_table_filter_destroy(&raw mut self.0) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(table_filter_test_support)]
    mod table_filter_tests {
        use super::super::TableFilterClass;
        use super::super::test_support::OwnedTableFilter;
        use crate::cpp::DUCKDB_VX_EXPR_TYPE;
        use crate::duckdb::ExtractedValue;
        use crate::duckdb::Value;

        #[test]
        fn empty_conjunctions_yield_empty_children() {
            let empty_and = OwnedTableFilter::conjunction_and(vec![]);
            assert!(matches!(
                empty_and.as_ref().as_class(),
                TableFilterClass::ConjunctionAnd(conj) if conj.children().count() == 0
            ));

            let empty_or = OwnedTableFilter::conjunction_or(vec![]);
            assert!(matches!(
                empty_or.as_ref().as_class(),
                TableFilterClass::ConjunctionOr(conj) if conj.children().count() == 0
            ));
        }

        #[test]
        fn dynamic_filter_data_latest_updates_on_set_value() {
            let (dyn_filter, dyn_data) = OwnedTableFilter::dynamic(
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN,
                &Value::from(100i32),
            );
            let TableFilterClass::Dynamic(dynamic) = dyn_filter.as_ref().as_class() else {
                panic!("expected Dynamic");
            };
            assert_eq!(
                dynamic.operator,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN
            );
            assert!(dynamic.data.latest().is_none());

            dyn_data.set_value(&Value::from(55i32));
            assert!(matches!(
                dynamic.data.latest().map(|v| v.extract()),
                Some(ExtractedValue::Integer(55))
            ));
        }
    }
}
