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
                    children: unsafe {
                        std::slice::from_raw_parts(out.children, out.children_count)
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
                    children: unsafe {
                        std::slice::from_raw_parts(out.children, out.children_count)
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
