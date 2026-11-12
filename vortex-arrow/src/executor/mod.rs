// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution of Vortex arrays into Arrow arrays.
//!
//! [`ExecuteArrowExt`] is the entry point for every export of a Vortex array to Arrow. Most targets
//! go to the canonical export, which maps the array to the Arrow type of the target. Only a target
//! with an Arrow extension name goes to the export plugins of the [`ArrowSession`] first. Each
//! submodule exports one family of Arrow types.
//!
//! [`ArrowSession`]: crate::ArrowSession

pub mod bool;
mod byte;
pub mod byte_view;
mod decimal;
mod dictionary;
mod fixed_size_list;
mod list;
mod list_view;
mod map;
pub mod null;
pub mod primitive;
mod run_end;
mod struct_;
mod temporal;
mod validity;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::types::*;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use tracing::debug;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::List;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::Id;

use crate::ArrowExport;
use crate::ArrowExportOptions;
use crate::executor::bool::to_arrow_bool;
use crate::executor::byte::to_arrow_byte_array;
use crate::executor::byte_view::to_arrow_byte_view;
use crate::executor::decimal::to_arrow_decimal;
use crate::executor::dictionary::to_arrow_dictionary;
use crate::executor::fixed_size_list::to_arrow_fixed_list;
use crate::executor::list::to_arrow_list;
use crate::executor::list_view::to_arrow_list_view;
use crate::executor::map::to_arrow_map;
use crate::executor::null::to_arrow_null;
use crate::executor::primitive::to_arrow_primitive;
use crate::executor::run_end::to_arrow_run_end;
use crate::executor::struct_::to_arrow_struct;
use crate::executor::temporal::to_arrow_date;
use crate::executor::temporal::to_arrow_time;
use crate::executor::temporal::to_arrow_timestamp;
use crate::ext_metadata::decode_arrow_ext_metadata;
use crate::session::ArrowSessionExt;
use crate::session::ensure_ext_metadata;

/// Execution of a Vortex array into an Arrow array, for every type that implements [`IntoArray`].
///
/// An export uses the [`ArrowSession`] of its [`ExecutionCtx`] for every decision: the export
/// plugins, the checks of the target, and the Arrow types of nested arrays. It takes no other
/// session, so one export can never mix the plugins of two sessions.
///
/// ```
/// # use arrow_array::Array;
/// # use vortex_array::IntoArray;
/// # use vortex_array::VortexSessionExecute;
/// # use vortex_array::arrays::VarBinViewArray;
/// use vortex_arrow::ExecuteArrowExt;
/// # use vortex_session::VortexSession;
///
/// let session = VortexSession::empty();
/// let array = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
/// let arrow = array.execute_arrow(None, &mut session.create_execution_ctx())?;
/// assert_eq!(arrow.len(), 2);
/// # Ok::<(), vortex_error::VortexError>(())
/// ```
///
/// [`ArrowSession`]: crate::ArrowSession
pub trait ExecuteArrowExt: IntoArray + Sized {
    /// Execute this array into an Arrow array of the type of `target`, with the default
    /// [`ArrowExportOptions`].
    ///
    /// A `target` with an Arrow extension name goes to the plugins registered for that name, in
    /// registration order. If none of them accepts the array, but the name is the Vortex ID of the
    /// array, the array exports as its storage (see [`ArrowSession`]).
    ///
    /// Without a `target`, the array exports to the field from
    /// [`ArrowSession::default_arrow_export_field`], so it fails for the same types.
    ///
    /// [`ArrowSession`]: crate::ArrowSession
    /// [`ArrowSession::default_arrow_export_field`]: crate::ArrowSession::default_arrow_export_field
    fn execute_arrow(
        self,
        target: Option<&Field>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow_with_options(target, &ArrowExportOptions::default(), ctx)
    }

    /// Execute this array into an Arrow array with `options`, which reach every nested array and
    /// every export plugin.
    ///
    /// Otherwise, this behaves like [`execute_arrow`](Self::execute_arrow).
    ///
    /// ```
    /// # use arrow_array::Array;
    /// # use vortex_array::IntoArray;
    /// # use vortex_array::VortexSessionExecute;
    /// # use vortex_array::arrays::VarBinViewArray;
    /// # use vortex_arrow::ArrowExportOptions;
    /// # use vortex_arrow::CompactBuffers;
    /// use vortex_arrow::ExecuteArrowExt;
    /// # use vortex_session::VortexSession;
    ///
    /// let session = VortexSession::empty();
    /// let array = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
    /// let options = ArrowExportOptions::default().with(CompactBuffers(false));
    /// let arrow =
    ///     array.execute_arrow_with_options(None, &options, &mut session.create_execution_ctx())?;
    /// assert_eq!(arrow.len(), 2);
    /// # Ok::<(), vortex_error::VortexError>(())
    /// ```
    fn execute_arrow_with_options(
        self,
        target: Option<&Field>,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        execute_arrow_array(self.into_array(), target, options, ctx)
    }
}

impl<T: IntoArray> ExecuteArrowExt for T {}

/// Export `array` to `target`, or to the default field of its dtype if there is no target.
fn execute_arrow_array(
    array: ArrayRef,
    target: Option<&Field>,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // NOTE(aduffy): this looks strange, but we do this to keep target_field as &Field so
    //  we can avoid cloning target when it is provided. It contains a HashMap internally that
    //  can be expensive to copy.
    let arrow_field;
    let target_field = match target {
        Some(field) => field,
        None => {
            arrow_field = ctx
                .session()
                .arrow()
                .default_arrow_export_field(array.dtype())?;
            &arrow_field
        }
    };

    if let DataType::Dictionary(codes, values) = target_field.data_type() {
        let values_field = target_field.clone().with_data_type(values.as_ref().clone());
        return to_arrow_dictionary(array, codes, &values_field, options, ctx);
    }

    if let Some(arrow_ext_name) = target_field.extension_type_name() {
        return execute_arrow_extension(array, arrow_ext_name, target_field, options, ctx);
    }

    execute_arrow_naive(array, target.map(Field::data_type), options, ctx)
}

/// Export `array` to `target`, whose Arrow extension name is `arrow_ext_name`, through the first
/// export plugin that accepts it, or as its storage if `arrow_ext_name` is its Vortex ID.
#[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
fn execute_arrow_extension(
    array: ArrayRef,
    arrow_ext_name: &str,
    target: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let arrow_ext_id = Id::new(arrow_ext_name);
    let exporters = ctx.session().arrow().exporters(&arrow_ext_id);

    // There can be multiple plugins that report support for a particular extension type. We try
    // them in order until one of them reports a successful conversion.
    let len = array.len();
    let mut current = array;

    for plugin in exporters.iter() {
        trace!(
            plugin = ?plugin,
            extension_name = arrow_ext_name,
            "probing plugin for converting Arrow array"
        );

        match plugin.execute_arrow(current, target, options, ctx)? {
            ArrowExport::Exported(arrow) => {
                vortex_ensure_eq!(
                    arrow.len(),
                    len,
                    "Arrow array length does not match Vortex array length after conversion to {:?}",
                    arrow
                );
                return Ok(arrow);
            }
            ArrowExport::Unsupported(array) => current = array,
        }
    }

    // The extension name can be an alias, which the dtype registry resolves to the Vortex ID of its
    // type, so the array can match either of them.
    let named_vortex_id = ctx.session().arrow().known_vortex_id(&arrow_ext_id);

    if let Some(ext_dtype) = current
        .dtype()
        .as_extension_opt()
        .filter(|ext_dtype| {
            ext_dtype.id() == arrow_ext_id || Some(ext_dtype.id()) == named_vortex_id
        })
        .cloned()
    {
        return execute_arrow_ext_storage(current, &ext_dtype, target, options, ctx);
    }

    // An extension array of another type must not export to a field that names this Vortex ID. A
    // plain array can, because a plugin such as `Vector` exports its storage to the same field that
    // it received.
    if let (Some(ext_dtype), Some(named_vortex_id)) =
        (current.dtype().as_extension_opt(), named_vortex_id)
    {
        vortex_bail!(
            "Arrow field {:?} names the extension type {named_vortex_id}, so the array must have that type, got {ext_dtype}",
            target.name()
        );
    }

    debug!(
        extension_id = arrow_ext_name,
        data_type = ?target.data_type(),
        "unsupported Arrow extension type encountered, falling back to naive execution"
    );

    execute_arrow_naive(current, Some(target.data_type()), options, ctx)
}

/// Export `array`, whose type is `ext_dtype`, to `target`, whose extension name is the Vortex ID of
/// `ext_dtype`. [`ArrowSession`](crate::ArrowSession) describes the checks on `target`.
fn execute_arrow_ext_storage(
    array: ArrayRef,
    ext_dtype: &ExtDTypeRef,
    target: &Field,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    ensure_ext_metadata(target, &decode_arrow_ext_metadata(target)?, ext_dtype)?;

    let target_dtype = ctx
        .session()
        .arrow()
        .from_arrow_datatype(target.data_type(), ext_dtype.nullability())?;

    // If the Arrow type of the target imports as the extension type itself, such as an Arrow
    // timestamp, export the array to it. The executor of that Arrow type then checks the parameters
    // of the type, such as the time unit of a timestamp.
    if target_dtype.eq_ignore_nullability(array.dtype()) {
        return execute_arrow_naive(array, Some(target.data_type()), options, ctx);
    }

    vortex_ensure!(
        eq_storage_layout(&target_dtype, ext_dtype.storage_dtype()),
        "Arrow field {:?} must have an Arrow type that imports as the storage of {ext_dtype}, got {}",
        target.name(),
        target.data_type()
    );

    let storage = array
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();

    execute_arrow_naive(storage, Some(target.data_type()), options, ctx)
}

/// Returns whether `lhs` and `rhs` have the same storage layout, which means that they are equal,
/// ignoring nullability, after every nested extension type is replaced by its storage type. The
/// types at the top level are compared as they are.
///
/// An explicit target may omit nested extension metadata or name an unregistered Vortex ID, so
/// export compares its inferred type with the storage layout. Each nested field is checked when
/// it is exported.
fn eq_storage_layout(lhs: &DType, rhs: &DType) -> bool {
    match (lhs, rhs) {
        (DType::List(lhs, _), DType::List(rhs, _)) => eq_nested_storage_layout(lhs, rhs),
        (DType::FixedSizeList(lhs, lhs_size, _), DType::FixedSizeList(rhs, rhs_size, _)) => {
            lhs_size == rhs_size && eq_nested_storage_layout(lhs, rhs)
        }
        (DType::Map(lhs, _), DType::Map(rhs, _)) => {
            lhs.keys_sorted() == rhs.keys_sorted()
                && eq_nested_storage_layout(&lhs.key_dtype(), &rhs.key_dtype())
                && eq_nested_storage_layout(&lhs.value_dtype(), &rhs.value_dtype())
        }
        (DType::Struct(lhs, _), DType::Struct(rhs, _)) => {
            lhs.names() == rhs.names()
                && lhs
                    .fields()
                    .zip(rhs.fields())
                    .all(|(lhs, rhs)| eq_nested_storage_layout(&lhs, &rhs))
        }
        _ => lhs.eq_ignore_nullability(rhs),
    }
}

/// [`eq_storage_layout`] for two nested types, which replaces their own extension types by their
/// storage types too.
fn eq_nested_storage_layout(lhs: &DType, rhs: &DType) -> bool {
    eq_storage_layout(innermost_storage_dtype(lhs), innermost_storage_dtype(rhs))
}

/// Returns `dtype` without its extension types, which can wrap each other.
fn innermost_storage_dtype(mut dtype: &DType) -> &DType {
    while let Some(ext_dtype) = dtype.as_extension_opt() {
        dtype = ext_dtype.storage_dtype();
    }

    dtype
}

/// Execute an arbitrary Vortex array into an Arrow array, dispatched by [`DataType`]. This pathway
/// is naive to any extension type information, and only seeks to map canonical Vortex arrays to
/// some target Arrow physical encoding.
///
/// Public callers should go through [`ExecuteArrowExt`] instead.
pub(crate) fn execute_arrow_naive(
    array: ArrayRef,
    data_type: Option<&DataType>,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let len = array.len();

    let resolved_type: DataType = match data_type {
        Some(dt) => dt.clone(),
        None => infer_nearest_arrow_type(&array, ctx)?,
    };

    let arrow = match &resolved_type {
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
        DataType::BinaryView => to_arrow_byte_view::<BinaryViewType>(array, options, ctx),
        DataType::Utf8View => to_arrow_byte_view::<StringViewType>(array, options, ctx),
        // TODO(joe): pass down preferred
        DataType::List(elements_field) => to_arrow_list::<i32>(array, elements_field, options, ctx),
        // TODO(joe): pass down preferred
        DataType::LargeList(elements_field) => {
            to_arrow_list::<i64>(array, elements_field, options, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::FixedSizeList(elements_field, list_size) => {
            to_arrow_fixed_list(array, *list_size, elements_field, options, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::ListView(elements_field) => {
            to_arrow_list_view::<i32>(array, elements_field, options, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::LargeListView(elements_field) => {
            to_arrow_list_view::<i64>(array, elements_field, options, ctx)
        }
        DataType::Struct(fields) => {
            let fields = if data_type.is_none() {
                None
            } else {
                Some(fields)
            };
            to_arrow_struct(array, fields, options, ctx)
        }
        // TODO(joe): pass down preferred
        DataType::Dictionary(codes_type, values_type) => to_arrow_dictionary(
            array.clone(),
            codes_type,
            &Field::new(
                "",
                values_type.as_ref().clone(),
                array.dtype().is_nullable(),
            ),
            options,
            ctx,
        ),
        dt @ DataType::Decimal32(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal64(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal128(..) => to_arrow_decimal(array, dt, ctx),
        dt @ DataType::Decimal256(..) => to_arrow_decimal(array, dt, ctx),
        // TODO(joe): pass down preferred
        DataType::RunEndEncoded(ends_type, values_type) => {
            to_arrow_run_end(array, ends_type.data_type(), values_type, options, ctx)
        }
        dt @ (DataType::Date32 | DataType::Date64) => to_arrow_date(array, dt, ctx),
        dt @ (DataType::Time32(_) | DataType::Time64(_)) => to_arrow_time(array, dt, ctx),
        dt @ DataType::Timestamp(..) => to_arrow_timestamp(array, dt, ctx),
        DataType::Map(entries_field, keys_sorted) => {
            to_arrow_map(array, entries_field, *keys_sorted, options, ctx)
        }
        DataType::FixedSizeBinary(_)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Union(..) => {
            vortex_bail!("Conversion to Arrow type {resolved_type} is not supported");
        }
    }?;

    vortex_ensure_eq!(
        arrow.len(),
        len,
        "Arrow array length does not match Vortex array length after conversion to {:?}",
        arrow
    );

    Ok(arrow)
}

/// Determine the preferred (cheapest) Arrow type for an array.
///
/// For most arrays, this returns the canonical Arrow type for the array's dtype.
/// However, some encodings have cheaper Arrow representations:
/// - `VarBinArray`: Uses `Utf8`/`Binary` (offset-based) instead of `Utf8View`/`BinaryView`
/// - `ListArray`: Uses `List` instead of `ListView`
fn infer_nearest_arrow_type(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DataType> {
    // VarBinArray: use offset-based Binary/Utf8 instead of View types
    if let Some(varbin) = array.as_opt::<VarBin>() {
        let offsets_ptype = PType::try_from(varbin.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);

        return Ok(match (varbin.dtype(), use_large) {
            (DType::Utf8(_), false) => DataType::Utf8,
            (DType::Utf8(_), true) => DataType::LargeUtf8,
            (DType::Binary(_), false) => DataType::Binary,
            (DType::Binary(_), true) => DataType::LargeBinary,
            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        });
    }

    // ListArray: use List with appropriate offset size
    if let Some(list) = array.as_opt::<List>() {
        let offsets_ptype = PType::try_from(list.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);
        // Recursively get the preferred field for elements, so extension elements keep the
        // `ARROW:extension:name` metadata their export plugin assigns.
        let field = FieldRef::new(infer_nearest_arrow_field(
            list.elements(),
            Field::LIST_FIELD_DEFAULT_NAME,
            ctx,
        )?);

        return Ok(if use_large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        });
    }

    // Everything else: defer to the session's canonical conversion, which additionally resolves
    // extension dtypes (including ones nested inside containers) through their export plugins.
    Ok(infer_nearest_arrow_field(array, "", ctx)?
        .data_type()
        .clone())
}

/// Determine the preferred (cheapest) Arrow [`Field`] for an array.
///
/// Unlike [`infer_nearest_arrow_type`] this preserves the Field-level `ARROW:extension:name`
/// metadata that export plugins assign, which a bare [`DataType`] cannot carry.
pub(crate) fn infer_nearest_arrow_field(
    array: &ArrayRef,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Field> {
    // Only the encodings with a cheaper-than-canonical Arrow type need the inference; every other
    // dtype goes through the session, which is the only thing that knows how to map extensions.
    if array.is::<VarBin>() || array.is::<List>() {
        let data_type = infer_nearest_arrow_type(array, ctx)?;
        return Ok(Field::new(name, data_type, array.dtype().is_nullable()));
    }
    ctx.session().arrow().to_arrow_field(name, array.dtype())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Array;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::BinaryArray;
    use arrow_array::Date32Array;
    use arrow_array::Float64Array;
    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_array::LargeStringArray;
    use arrow_array::TimestampMicrosecondArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::TimeUnit;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_session::VortexSession;

    use crate::ExecuteArrowExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn utf8() -> ArrayRef {
        VarBinViewArray::from_iter_str(["a", "bb"]).into_array()
    }

    fn primitive() -> ArrayRef {
        PrimitiveArray::from_iter([1i32, 2]).into_array()
    }

    fn boolean() -> ArrayRef {
        BoolArray::from_iter([true, false]).into_array()
    }

    fn list_target() -> DataType {
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
    }

    fn map_target() -> DataType {
        let entries = Field::new_struct(
            "entries",
            vec![
                Field::new("key", DataType::Int32, false),
                Field::new("value", DataType::Int32, true),
            ],
            false,
        );
        DataType::Map(Arc::new(entries), false)
    }

    /// Must error rather than panic inside `execute::<T>`.
    #[rstest]
    #[case::bool_from_utf8(utf8(), DataType::Boolean)]
    #[case::bool_from_primitive(primitive(), DataType::Boolean)]
    #[case::null_from_utf8(utf8(), DataType::Null)]
    #[case::null_from_primitive(primitive(), DataType::Null)]
    #[case::decimal_from_utf8(utf8(), DataType::Decimal128(10, 2))]
    #[case::decimal_from_primitive(primitive(), DataType::Decimal128(10, 2))]
    #[case::list_from_utf8(utf8(), list_target())]
    #[case::list_from_primitive(primitive(), list_target())]
    #[case::list_view_from_primitive(
        primitive(),
        DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true)))
    )]
    #[case::fixed_size_list_from_utf8(
        utf8(),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2)
    )]
    #[case::map_from_primitive(primitive(), map_target())]
    #[case::byte_from_primitive(primitive(), DataType::Utf8)]
    #[case::byte_from_bool(boolean(), DataType::Binary)]
    fn incompatible_target_returns_error(#[case] array: ArrayRef, #[case] target: DataType) {
        let mut ctx = SESSION.create_execution_ctx();
        let field = Field::new("f", target.clone(), array.dtype().is_nullable());

        let result = array.execute_arrow(Some(&field), &mut ctx);

        assert!(
            result.is_err(),
            "expected an error exporting to {target:?}, got Ok"
        );
    }

    /// Cross-class conversions that are genuinely supported must keep working.
    #[rstest]
    #[case::bool_to_int32(boolean(), DataType::Int32, Arc::new(Int32Array::from(vec![1, 0])))]
    #[case::bool_to_float64(boolean(), DataType::Float64, Arc::new(Float64Array::from(vec![1., 0.])))]
    #[case::primitive_to_int64(primitive(), DataType::Int64, Arc::new(Int64Array::from(vec![1, 2])))]
    #[case::primitive_to_date32(primitive(), DataType::Date32, Arc::new(Date32Array::from(vec![1, 2])))]
    #[case::primitive_to_timestamp(primitive(), DataType::Timestamp(TimeUnit::Microsecond, None), Arc::new(TimestampMicrosecondArray::from(vec![1, 2])))]
    #[case::utf8_to_binary(utf8(), DataType::Binary, Arc::new(BinaryArray::from(vec![b"a".as_slice(), b"bb".as_slice()])))]
    #[case::utf8_to_large_utf8(utf8(), DataType::LargeUtf8, Arc::new(LargeStringArray::from(vec!["a", "bb"])))]
    fn supported_cross_class_target_still_works(
        #[case] array: ArrayRef,
        #[case] target: DataType,
        #[case] expected: ArrowArrayRef,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let field = Field::new("f", target, array.dtype().is_nullable());

        let result = array.execute_arrow(Some(&field), &mut ctx);

        assert_eq!(result.unwrap().to_data(), expected.to_data());
    }
}
