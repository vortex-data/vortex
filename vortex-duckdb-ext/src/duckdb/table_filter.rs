use std::ffi::CStr;
use std::fmt::{Debug, Formatter};
use std::ptr;

use crate::duckdb::Value;
use crate::{cpp, wrapper};

wrapper!(TableFilterSet, cpp::duckdb_vx_table_filter_set, |_| {});

impl TableFilterSet {
    pub fn get(&self, column_idx: u64) -> Option<TableFilter> {
        let ptr = unsafe { cpp::duckdb_vx_table_filter_set_get(self.as_ptr(), column_idx) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilter::from_ptr(ptr) })
        }
    }
}

wrapper!(TableFilter, cpp::duckdb_vx_table_filter, |_| {});

impl TableFilter {
    pub fn as_class(&self) -> Option<TableFilterClass> {
        Some(
            match unsafe { cpp::duckdb_vx_table_filter_get_type(self.as_ptr()) } {
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_CONSTANT_COMPARISON => {
                    let mut out = cpp::duckdb_vx_table_filter_constant {
                        value: ptr::null_mut(),
                        comparison_type: cpp::DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID,
                    };
                    unsafe { cpp::duckdb_vx_table_filter_get_constant(self.as_ptr(), &raw mut out) };

                    TableFilterClass::ConstantComparison(ConstantComparison {
                        value: unsafe { Value::from_ptr(out.value) },
                        operator: out.comparison_type,
                    })
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IS_NULL => {
                    TableFilterClass::IsNull
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IS_NOT_NULL => {
                    TableFilterClass::IsNotNull
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_CONJUNCTION_OR => {
                    let mut out = cpp::duckdb_vx_table_filter_conjunction {
                        children: ptr::null_mut(),
                        children_count: 0,
                    };
                    unsafe { cpp::duckdb_vx_table_filter_get_conjunction_or(self.as_ptr(), &raw mut out) };

                    TableFilterClass::ConjunctionOr(Conjunction {
                        children: unsafe {
                            std::slice::from_raw_parts(out.children, out.children_count)
                        },
                    })
                },
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
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_STRUCT_EXTRACT => {
                    return None;
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_OPTIONAL_FILTER => {
                    return None;
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_IN_FILTER => {
                    return None;
                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_DYNAMIC_FILTER => {
                    let filter_data = unsafe {
                        cpp::duckdb_vx_table_filter_get_dynamic(self.as_ptr())
                    };
                    TableFilterClass::Dynamic(unsafe { DynamicFilterData::from_ptr(filter_data) })

                },
                cpp::DUCKDB_VX_TABLE_FILTER_TYPE::DUCKDB_VX_TABLE_FILTER_TYPE_EXPRESSION_FILTER => {
                    return None;
                },
            },
        )
    }
}

impl Debug for TableFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_vx_table_filter_to_debug_string(self.as_ptr()) };
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let result = write!(f, "{}", cstr.to_string_lossy());
        unsafe { cpp::duckdb_free(ptr.cast_mut().cast()) };
        result
    }
}

pub enum TableFilterClass<'a> {
    ConstantComparison(ConstantComparison),
    IsNull,
    IsNotNull,
    ConjunctionOr(Conjunction<'a>),
    ConjunctionAnd(Conjunction<'a>),
    Dynamic(DynamicFilterData),
}

pub struct ConstantComparison {
    pub value: Value,
    pub operator: cpp::DUCKDB_VX_EXPR_TYPE,
}

pub struct Conjunction<'a> {
    children: &'a [cpp::duckdb_vx_table_filter],
}

impl Conjunction<'_> {
    pub fn children(&self) -> impl Iterator<Item = TableFilter> {
        self.children
            .iter()
            .map(|&child| unsafe { TableFilter::from_ptr(child) })
    }
}

wrapper!(
    /// A handle to mutable dynamic filter data.
    DynamicFilterData,
    cpp::duckdb_vx_dynamic_filter_data,
    |_| {}
);
