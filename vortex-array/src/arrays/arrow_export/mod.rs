// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution node for Arrow export.
//!
//! `ArrowExportArray` wraps a child array and a target Arrow `DataType`. It participates
//! in the execution loop so that encodings can register parent kernels for `ArrowExport`
//! to intercept and produce `NativeArrowArray` directly.

use std::hash::Hasher;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::types::*;
use arrow_schema::DataType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::Extension;
use crate::arrays::NativeArrow;
use crate::arrays::NativeArrowArray;
use crate::arrow::executor::bool::to_arrow_bool;
use crate::arrow::executor::byte::to_arrow_byte_array;
use crate::arrow::executor::byte_view::to_arrow_byte_view;
use crate::arrow::executor::decimal::to_arrow_decimal;
use crate::arrow::executor::dictionary::to_arrow_dictionary;
use crate::arrow::executor::fixed_size_list::to_arrow_fixed_list;
use crate::arrow::executor::list::to_arrow_list;
use crate::arrow::executor::list_view::to_arrow_list_view;
use crate::arrow::executor::null::to_arrow_null;
use crate::arrow::executor::primitive::to_arrow_primitive;
use crate::arrow::executor::run_end::to_arrow_run_end;
use crate::arrow::executor::struct_::to_arrow_struct;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

vtable!(ArrowExport);

#[derive(Debug)]
pub struct ArrowExport;

impl ArrowExport {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.arrow_export");
}

/// Execution node carrying a child array and a target Arrow `DataType`.
///
/// During execution, encodings can register parent kernels for `ArrowExport`
/// to intercept the export and produce `NativeArrowArray` directly (e.g., Dict → DictionaryArray).
#[derive(Clone, Debug)]
pub struct ArrowExportArray {
    child: ArrayRef,
    target: DataType,
    dtype: DType,
    stats_set: ArrayStats,
}

impl ArrowExportArray {
    /// Create a new `ArrowExportArray` wrapping the given child and target Arrow type.
    pub fn new(child: ArrayRef, target: DataType) -> Self {
        let dtype = child.dtype().clone();
        Self {
            child,
            target,
            dtype,
            stats_set: Default::default(),
        }
    }

    /// Returns the target Arrow `DataType`.
    pub fn target(&self) -> &DataType {
        &self.target
    }

    /// Returns the child array.
    pub fn child_array(&self) -> &ArrayRef {
        &self.child
    }
}

impl VTable for ArrowExport {
    type Array = ArrowExportArray;
    type Metadata = ();
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &ArrowExportArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &ArrowExportArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ArrowExportArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(_array: &ArrowExportArray, _state: &mut H, _precision: Precision) {
        vortex_panic!("ArrowExportArray is transient and does not support hashing")
    }

    fn array_eq(
        _array: &ArrowExportArray,
        _other: &ArrowExportArray,
        _precision: Precision,
    ) -> bool {
        vortex_panic!("ArrowExportArray is transient and does not support equality")
    }

    fn nbuffers(_array: &ArrowExportArray) -> usize {
        0
    }

    fn buffer(_array: &ArrowExportArray, idx: usize) -> BufferHandle {
        vortex_panic!("ArrowExportArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ArrowExportArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &ArrowExportArray) -> usize {
        1
    }

    fn child(array: &ArrowExportArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.child.clone(),
            _ => vortex_panic!("ArrowExportArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &ArrowExportArray, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("ArrowExportArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &ArrowExportArray) -> VortexResult<Self::Metadata> {
        Ok(())
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("ArrowExportArray cannot be deserialized")
    }

    fn build(
        _dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ArrowExportArray> {
        vortex_bail!("ArrowExportArray cannot be built from components")
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "ArrowExportArray expects 1 child, got {}",
            children.len()
        );
        array.child = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("ArrowExportArray: expected 1 child"))?;
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        // If child is already NativeArrow, we're done.
        if array.child.is::<NativeArrow>() {
            return Ok(ExecutionStep::Done(array.child.clone()));
        }

        // If child is canonical, convert via DataType dispatch.
        if AnyCanonical::matches(array.child.as_ref()) {
            // Extension arrays need special handling — delegate to the ExtDType.
            if let Some(ext_array) = array.child.as_opt::<Extension>() {
                let ext_dtype = ext_array.ext_dtype().clone();
                let storage = ext_array.storage_array().clone();
                let arrow = ext_dtype.to_arrow_array(storage, &array.target, ctx)?;
                let native = NativeArrowArray::new(arrow, array.dtype.clone());
                return Ok(ExecutionStep::Done(native.into_array()));
            }

            let arrow = datatype_dispatch(array.child.clone(), &array.target, ctx)?;
            let native = NativeArrowArray::new(arrow, array.dtype.clone());
            return Ok(ExecutionStep::Done(native.into_array()));
        }

        // Otherwise, ask the scheduler to execute the child to canonical form.
        Ok(ExecutionStep::execute_child::<AnyCanonical>(0))
    }
}

/// DataType-based dispatch for converting a canonical Vortex array to Arrow.
///
/// This is the same logic that was previously in `execute_arrow()` (the DataType match block),
/// extracted here so it can be used by `ArrowExportArray::execute()`.
pub(crate) fn datatype_dispatch(
    array: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    match data_type {
        DataType::Null => to_arrow_null(array, ctx),
        DataType::Boolean => to_arrow_bool(array, ctx),
        DataType::Int8 => to_arrow_primitive::<Int8Type>(array, ctx),
        DataType::Int16 => to_arrow_primitive::<Int16Type>(array, ctx),
        DataType::Int32 => to_arrow_primitive::<Int32Type>(array, ctx),
        DataType::Int64 => to_arrow_primitive::<Int64Type>(array, ctx),
        DataType::UInt8 => to_arrow_primitive::<UInt8Type>(array, ctx),
        DataType::UInt16 => to_arrow_primitive::<UInt16Type>(array, ctx),
        DataType::UInt32 => to_arrow_primitive::<UInt32Type>(array, ctx),
        DataType::UInt64 => to_arrow_primitive::<UInt64Type>(array, ctx),
        DataType::Float16 => to_arrow_primitive::<Float16Type>(array, ctx),
        DataType::Float32 => to_arrow_primitive::<Float32Type>(array, ctx),
        DataType::Float64 => to_arrow_primitive::<Float64Type>(array, ctx),
        DataType::Binary => to_arrow_byte_array::<BinaryType>(array, ctx),
        DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(array, ctx),
        DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(array, ctx),
        DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(array, ctx),
        DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(array, ctx),
        DataType::Utf8View => to_arrow_byte_view::<StringViewType>(array, ctx),
        DataType::List(elements_field) => to_arrow_list::<i32>(array, elements_field, ctx),
        DataType::LargeList(elements_field) => to_arrow_list::<i64>(array, elements_field, ctx),
        DataType::FixedSizeList(elements_field, list_size) => {
            to_arrow_fixed_list(array, *list_size, elements_field, ctx)
        }
        DataType::ListView(elements_field) => to_arrow_list_view::<i32>(array, elements_field, ctx),
        DataType::LargeListView(elements_field) => {
            to_arrow_list_view::<i64>(array, elements_field, ctx)
        }
        DataType::Struct(fields) => to_arrow_struct(array, Some(fields), ctx),
        DataType::Dictionary(codes_type, values_type) => {
            to_arrow_dictionary(array, codes_type, values_type, ctx)
        }
        dt @ DataType::Decimal32(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal64(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal128(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal256(..) => to_arrow_decimal(array, dt, ctx),
        DataType::RunEndEncoded(ends_type, values_type) => {
            to_arrow_run_end(array, ends_type.data_type(), values_type, ctx)
        }
        DataType::Timestamp(..)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::FixedSizeBinary(_)
        | DataType::Map(..)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Union(..) => {
            vortex_bail!("Conversion to Arrow type {data_type} is not supported");
        }
    }
}
