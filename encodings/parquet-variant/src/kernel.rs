// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::FieldRef;
use parquet_variant::VariantPath as PqVariantPath;
use parquet_variant::VariantPathElement as PqVariantPathElement;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::shred_variant;
use parquet_variant_compute::variant_get as arrow_variant_get;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_json::Json;
use vortex_json::JsonToVariant;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;
use crate::arrow::merge_storage_chunks;
use crate::compute::AllNonDistinctParquetVariant;
use crate::extract::extract;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Filter.id(),
        ParquetVariant,
        FilterExecuteAdaptor(ParquetVariant),
    );
    kernels.register_execute_parent_kernel(
        Slice.id(),
        ParquetVariant,
        SliceExecuteAdaptor(ParquetVariant),
    );
    kernels.register_execute_parent_kernel(
        Dict.id(),
        ParquetVariant,
        TakeExecuteAdaptor(ParquetVariant),
    );
    kernels.register_execute_parent_kernel(VariantGet.id(), ParquetVariant, VariantGetKernel);
    kernels.register_execute_parent_kernel(Chunked.id(), ParquetVariant, ChunkedStorageKernel);
    kernels.register_execute_parent_kernel(
        JsonToVariant.id(),
        Extension,
        JsonExtensionToVariantKernel,
    );
    let aggregates = session.aggregate_fns();
    aggregates.register_aggregate_kernel(
        Struct.id(),
        Some(AllNonDistinct.id()),
        &AllNonDistinctParquetVariant,
    );
}

/// Executes chunks of Parquet Variant storage to one storage array with chunked children.
///
/// A chunked Variant otherwise canonicalizes to a Variant over chunked storage, which keeps every
/// chunk a separate array: to compress on its own, and to deserialize as its own subtree.
#[derive(Default, Debug)]
struct ChunkedStorageKernel;

impl ExecuteParentKernel<ParquetVariant> for ChunkedStorageKernel {
    type Parent = Chunked;

    fn execute_parent(
        &self,
        array: ArrayView<'_, ParquetVariant>,
        parent: ArrayView<'_, Chunked>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only plain storage merges. A shredded child has to be made canonical chunk by chunk,
        // which rebuilds it from the layout of its own chunk, so those chunks stay apart, as do
        // chunks of other encodings and chunks that disagree on having a value.
        let mergeable = parent.iter_chunks().all(|chunk| {
            chunk.as_opt::<ParquetVariant>().is_some_and(|chunk| {
                chunk.typed_value().is_none() && chunk.value().is_some() == array.value().is_some()
            })
        });
        if !mergeable {
            return Ok(None);
        }
        merge_storage_chunks(&parent.chunks(), parent.dtype().nullability()).map(Some)
    }
}

#[derive(Default, Debug)]
struct VariantGetKernel;

impl ExecuteParentKernel<ParquetVariant> for VariantGetKernel {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, ParquetVariant>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        // Residual storage without a value in any row holds no path, as for the rows of a
        // shredded field that the typed column does not cover. Answer from the validity of the
        // value alone, without decoding the metadata. This holds for any requested dtype,
        // Variant included: a missing path is a null of it.
        if array.typed_value().is_none()
            && let Some(value) = array.value()
            && value
                .validity()?
                .execute_mask(value.len(), ctx)?
                .all_false()
        {
            return Ok(Some(
                ConstantArray::new(Scalar::null(parent.dtype().clone()), array.len()).into_array(),
            ));
        }

        // A string or an integer under object fields of unshredded storage is read from the
        // encoded bytes in place. Whatever that does not cover falls through to the kernel.
        if let Some(extracted) = extract_in_place(&array, &parent, ctx)? {
            return Ok(Some(extracted));
        }

        let arrow_variant = array.to_arrow(&ArrowExportOptions::default(), ctx)?;
        let arrow_input: ArrowArrayRef = Arc::new(arrow_variant.into_inner());
        let session = ctx.session().clone();
        let as_type = to_arrow_as_type(parent.options.dtype(), &session.arrow())?;
        let get_options =
            GetOptions::new_with_path(to_parquet_variant_path(parent.options.path())?)
                .with_as_type(as_type.clone());

        let arrow_output = arrow_variant_get(&arrow_input, get_options)?;
        let output = if parent.options.dtype().is_none_or(DType::is_variant) {
            let arrow_variant_output = ArrowVariantArray::try_new(arrow_output.as_ref())?;
            ParquetVariant::from_arrow_variant_nullable(
                &arrow_variant_output,
                &ctx.session().arrow(),
            )?
        } else {
            // Import through the same `as_type` field the cast targeted, so an extension target
            // dtype comes back as that extension rather than as its bare storage type.
            let as_type = as_type.ok_or_else(|| {
                vortex_err!("a non-variant target dtype must produce an as_type field")
            })?;
            session.arrow().from_arrow_array(arrow_output, &as_type)?
        };

        vortex_ensure_eq!(
            output.dtype(),
            parent.dtype(),
            "VariantGet output dtype must match parent dtype"
        );
        Ok(Some(output))
    }
}

/// Extracts with [`extract`], if it covers the extraction that `parent` asks of `array`.
fn extract_in_place(
    array: &ArrayView<'_, ParquetVariant>,
    parent: &ScalarFnArrayView<'_, VariantGet>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(dtype), Some(value), None) =
        (parent.options.dtype(), array.value(), array.typed_value())
    else {
        return Ok(None);
    };
    if !matches!(dtype, DType::Utf8(_) | DType::Primitive(PType::I64, _)) {
        return Ok(None);
    }
    let Some(path) = parent
        .options
        .path()
        .elements()
        .iter()
        .map(|element| match element {
            VariantPathElement::Field(name) => Some(name.as_ref()),
            VariantPathElement::Index(_) => None,
        })
        .collect::<Option<Vec<&str>>>()
    else {
        return Ok(None);
    };

    let len = array.len();
    let valid = &array.parquet_variant_validity().execute_mask(len, ctx)?
        & &value.validity()?.execute_mask(len, ctx)?;
    let metadata = array.metadata().clone().execute::<VarBinViewArray>(ctx)?;
    let value = value.clone().execute::<VarBinViewArray>(ctx)?;
    let Some(extracted) = extract(&metadata, &value, &valid, &path, dtype) else {
        return Ok(None);
    };
    extracted.cast(parent.dtype().clone()).map(Some)
}

/// Performs the [`JsonToVariant`] conversion (and optional shredding) over JSON string storage.
///
/// `JsonToVariant`'s definition lives in `vortex-json`; the registered `execute_parent` kernels
/// delegate here to do the actual JSON parsing and optional shredding using
/// `parquet_variant_compute`, producing a [`ParquetVariant`] array. `strings` is the JSON
/// extension's storage array; it is executed to Arrow and parsed. Nullability of the result follows
/// `parent.dtype()`, which equals the input's nullability.
fn json_strings_to_variant(
    strings: ArrayRef,
    parent: ScalarFnArrayView<'_, JsonToVariant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = parent.dtype().is_nullable();
    let session = ctx.session().clone();
    let arrow_strings = strings.execute_arrow(None, ctx)?;
    // Any row that fails to parse as JSON fails the whole conversion.
    let arrow_variant = parquet_variant_compute::json_to_variant(&arrow_strings)?;

    let arrow_variant = if parent.options.shredding().is_empty() {
        arrow_variant
    } else {
        let mut builder = ShreddedSchemaBuilder::new();
        for (path, dtype) in parent.options.shredding().fields() {
            let field: FieldRef = Arc::new(session.arrow().to_arrow_field("shredded", dtype)?);
            builder = builder.with_path(to_parquet_variant_path(path)?, field)?;
        }
        shred_variant(&arrow_variant, &builder.build())?
    };

    if nullable {
        ParquetVariant::from_arrow_variant_nullable(&arrow_variant, &session.arrow())
    } else {
        ParquetVariant::from_arrow_variant(&arrow_variant, &session.arrow())
    }
}

/// Builds Parquet Variant arrays for [`JsonToVariant`] over a [`Json`] extension input.
///
/// This kernel unwraps the extension's string storage and runs the JSON conversion. It is keyed on
/// the shared extension encoding, so it declines any non-`Json` extension.
#[derive(Default, Debug)]
struct JsonExtensionToVariantKernel;

impl ExecuteParentKernel<Extension> for JsonExtensionToVariantKernel {
    type Parent = ExactScalarFn<JsonToVariant>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Extension>,
        parent: ScalarFnArrayView<'_, JsonToVariant>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 || !array.ext_dtype().is::<Json>() {
            return Ok(None);
        }
        json_strings_to_variant(array.storage_array().clone(), parent, ctx).map(Some)
    }
}

pub(crate) fn to_parquet_variant_path(path: &VariantPath) -> VortexResult<PqVariantPath<'static>> {
    path.elements()
        .iter()
        .map(|element| match element {
            VariantPathElement::Field(name) => Ok(PqVariantPathElement::field(Cow::Owned(
                name.as_ref().to_owned(),
            ))),
            VariantPathElement::Index(index) => {
                let index = usize::try_from(*index)
                    .map_err(|_| vortex_err!("VariantGet path index {index} is too large"))?;
                Ok(PqVariantPathElement::index(index))
            }
        })
        .collect::<VortexResult<Vec<_>>>()
        .map(PqVariantPath::new)
}

/// The Arrow field `variant_get` should cast to for a requested Vortex `dtype`.
///
/// Built as a [`Field`](arrow_schema::Field) rather than a bare `DataType` so an extension target
/// keeps its `ARROW:extension:name`; the result is nullable because an absent path yields null.
fn to_arrow_as_type(
    dtype: Option<&DType>,
    session: &ArrowSession,
) -> VortexResult<Option<FieldRef>> {
    match dtype {
        Some(dtype) if !dtype.is_variant() => Ok(Some(Arc::new(
            session
                .to_arrow_field("variant_get", dtype)?
                .with_nullable(true),
        ))),
        Some(_) | None => Ok(None),
    }
}

impl SliceKernel for ParquetVariant {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.slice(range.clone())?;
        let metadata = array.metadata().slice(range.clone())?;
        let value = array.value().map(|v| v.slice(range.clone())).transpose()?;
        let typed_value = array.typed_value().map(|tv| tv.slice(range)).transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

impl FilterKernel for ParquetVariant {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.filter(mask)?;
        let metadata = array.metadata().filter(mask.clone())?;
        let value = array.value().map(|v| v.filter(mask.clone())).transpose()?;
        let typed_value = array
            .typed_value()
            .map(|tv| tv.filter(mask.clone()))
            .transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

impl TakeExecute for ParquetVariant {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.take(indices)?;
        let metadata = array.metadata().take(indices.clone())?;
        let value = array.value().map(|v| v.take(indices.clone())).transpose()?;
        let typed_value = array
            .typed_value()
            .map(|tv| tv.take(indices.clone()))
            .transpose()?;
        Ok(Some(
            ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Array as ArrowArray;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::ShreddedSchemaBuilder;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use parquet_variant_compute::json_to_variant;
    use parquet_variant_compute::shred_variant;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray as VortexStructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::arrays::variant::VariantArraySlotsExt;
    use vortex_array::arrays::variant::decode_unshredded_storage;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar_is_null;
    use vortex_array::dtype::DType as VortexDType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::expr::variant_pack;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
    use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::to_arrow_as_type;
    use crate::ParquetVariant;
    use crate::ParquetVariantArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        vortex_uuid::initialize(&session);
        session
    });

    /// `variant_get`'s cast target must be a `Field` carrying `ARROW:extension:name`, so an
    /// extension target dtype survives both the cast and the import back into Vortex. A bare
    /// `DataType` would silently degrade the target to its storage type.
    #[test]
    fn to_arrow_as_type_keeps_extension_metadata() -> VortexResult<()> {
        use arrow_schema::extension::ExtensionType;
        use arrow_schema::extension::Uuid as ArrowUuid;
        use vortex_array::dtype::extension::ExtDType;
        use vortex_uuid::Uuid;
        use vortex_uuid::UuidMetadata;

        let storage = VortexDType::FixedSizeList(
            Arc::new(VortexDType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            Nullability::NonNullable,
        );
        let uuid = VortexDType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage)?.erased(),
        );

        let arrow = SESSION.arrow();
        let as_type = to_arrow_as_type(Some(&uuid), &arrow)?
            .ok_or_else(|| vortex_err!("expected an as_type field"))?;

        assert_eq!(as_type.data_type(), &DataType::FixedSizeBinary(16));
        assert_eq!(as_type.extension_type_name(), Some(ArrowUuid::NAME));
        // An absent path yields null, so the cast target must stay nullable.
        assert!(as_type.is_nullable());

        // And the field round-trips back to the extension dtype the caller asked for.
        assert_eq!(arrow.from_arrow_field(&as_type)?, uuid.as_nullable());
        Ok(())
    }

    /// A non-extension target stays a plain nullable field.
    #[test]
    fn to_arrow_as_type_plain_dtype() -> VortexResult<()> {
        let arrow = SESSION.arrow();
        let as_type = to_arrow_as_type(
            Some(&VortexDType::Primitive(
                PType::I32,
                Nullability::NonNullable,
            )),
            &arrow,
        )?
        .ok_or_else(|| vortex_err!("expected an as_type field"))?;
        assert_eq!(as_type.data_type(), &DataType::Int32);
        assert_eq!(as_type.extension_type_name(), None);
        assert!(as_type.is_nullable());
        Ok(())
    }

    /// Variant (and absent) targets ask for no cast at all.
    #[test]
    fn to_arrow_as_type_variant_targets_are_none() -> VortexResult<()> {
        let arrow = SESSION.arrow();
        assert!(
            to_arrow_as_type(Some(&VortexDType::Variant(Nullability::Nullable)), &arrow)?.is_none()
        );
        assert!(to_arrow_as_type(None, &arrow)?.is_none());
        Ok(())
    }

    fn make_unshredded_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        ParquetVariant::from_arrow_variant(&builder.build(), &SESSION.arrow())
    }

    fn make_nullable_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        let inner = builder.build().into_inner();

        let null_struct = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false, true, false])),
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&null_struct)?;
        ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())
    }

    fn make_unshredded_json_array(values: Vec<Option<&str>>) -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(values));
        let arrow_variant = json_to_variant(&json)?;
        ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())
    }

    fn parse_path(path: &str) -> VortexResult<VariantPath> {
        if path.is_empty() || path == "$" {
            return Ok(VariantPath::root());
        }

        let mut elements = Vec::new();
        let mut pos = usize::from(path.as_bytes().first() == Some(&b'$'));
        if pos == 1
            && path
                .as_bytes()
                .get(pos)
                .is_some_and(|byte| !matches!(byte, b'.' | b'['))
        {
            vortex_bail!("Invalid Variant path {path:?}: expected '.' or '[' after '$'");
        }

        while pos < path.len() {
            match path.as_bytes()[pos] {
                b'.' => {
                    pos += 1;
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                b'[' => {
                    let (index, next_pos) = parse_index(path, pos + 1)?;
                    elements.push(VariantPathElement::index(index));
                    pos = next_pos;
                }
                _ if pos == 0 => {
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                _ => {
                    vortex_bail!("Invalid Variant path {path:?}: expected '.', '[', or end of path")
                }
            }
        }

        Ok(VariantPath::new(elements))
    }

    fn parse_field(path: &str, start: usize) -> VortexResult<(&str, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected field name"
        );
        Ok((&path[start..pos], pos))
    }

    fn parse_index(path: &str, start: usize) -> VortexResult<(u64, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected list index"
        );
        vortex_ensure!(
            path.as_bytes().get(pos) == Some(&b']'),
            "Invalid Variant path {path:?}: expected closing ']'"
        );
        let index = path[start..pos]
            .parse()
            .map_err(|_| vortex_err!("Invalid Variant path {path:?}: list index is too large"))?;
        Ok((index, pos + 1))
    }

    fn execute_variant_get(
        array: ArrayRef,
        path: &str,
        dtype: Option<VortexDType>,
    ) -> VortexResult<ArrayRef> {
        let expr = variant_get(root(), parse_path(path)?, dtype);
        array
            .apply(&expr)?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    macro_rules! assert_rows_eq {
        ($actual:expr, $expected:expr, [$($expected_idx:expr),* $(,)?]) => {{
            let actual = $actual;
            let expected = $expected;
            let expected_rows = [$($expected_idx),*];
            assert_eq!(actual.len(), expected_rows.len());
            assert_eq!(actual.dtype(), expected.dtype());

            let mut ctx = SESSION.create_execution_ctx();
            for (actual_idx, expected_idx) in expected_rows.into_iter().enumerate() {
                assert_eq!(
                    actual.execute_scalar(actual_idx, &mut ctx)?,
                    expected.execute_scalar(expected_idx, &mut ctx)?,
                    "row {actual_idx} should match source row {expected_idx}",
                );
            }
        }};
    }

    macro_rules! assert_nulls {
        ($array:expr, [$($is_null:expr),* $(,)?]) => {{
            let array = $array;
            let expected = [$($is_null),*];
            assert_eq!(array.len(), expected.len());

            let mut ctx = SESSION.create_execution_ctx();
            for (idx, is_null) in expected.into_iter().enumerate() {
                assert_eq!(
                    array.execute_scalar(idx, &mut ctx)?.is_null(),
                    is_null,
                    "row {idx} nullness mismatch",
                );
            }
        }};
    }

    #[test]
    fn test_slice_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let sliced = arr.slice(1..3)?;

        assert_rows_eq!(&sliced, &arr, [1, 2]);
        Ok(())
    }

    #[test]
    fn test_slice_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        let sliced = arr.slice(0..3)?;

        assert_nulls!(&sliced, [false, true, false]);
        assert_rows_eq!(&sliced, &arr, [0, 1, 2]);
        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let mask = Mask::from_iter([true, false, true, false]);
        let filtered = arr.filter(mask)?;

        assert_rows_eq!(&filtered, &arr, [0, 2]);
        Ok(())
    }

    #[test]
    fn test_filter_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Keep rows 0 (valid), 1 (null), 3 (null)
        let mask = Mask::from_iter([true, true, false, true]);
        let filtered = arr.filter(mask)?;

        assert_nulls!(&filtered, [false, true, true]);
        assert_rows_eq!(&filtered, &arr, [0, 1, 3]);
        Ok(())
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let indices = PrimitiveArray::from_iter([2u64, 0, 3]);
        let taken = arr.take(indices.into_array())?;

        assert_rows_eq!(&taken, &arr, [2, 0, 3]);
        Ok(())
    }

    #[test]
    fn test_take_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Take: valid (0), null (1), null (3), valid (2)
        let indices = PrimitiveArray::from_iter([0u64, 1, 3, 2]);
        let taken = arr.take(indices.into_array())?;

        assert_nulls!(&taken, [false, true, true, false]);
        assert_rows_eq!(&taken, &arr, [0, 1, 3, 2]);
        Ok(())
    }

    #[rstest]
    #[case::field(
        "$.a",
        vec![
            Some(r#"{"a": 1}"#),
            None,
            Some(r#"{"a": null}"#),
            Some(r#"{"a": "wrong"}"#),
            Some(r#"{"b": 2}"#),
        ],
        vec![Some(1), None, None, None, None],
    )]
    #[case::list_index(
        "$.items[1]",
        vec![
            Some(r#"{"items": [10, 20]}"#),
            Some(r#"{"items": []}"#),
            Some(r#"{"items": ["x", 7]}"#),
        ],
        vec![Some(20), None, Some(7)],
    )]
    fn test_variant_get_unshredded_as_i32(
        #[case] path: &str,
        #[case] rows: Vec<Option<&str>>,
        #[case] expected: Vec<Option<i32>>,
    ) -> VortexResult<()> {
        let arr = make_unshredded_json_array(rows)?;
        let result = execute_variant_get(
            arr,
            path,
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_eq!(
            result.dtype(),
            &VortexDType::Primitive(PType::I32, Nullability::Nullable)
        );
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_variant_get_unshredded_field_as_variant() -> VortexResult<()> {
        let arr = make_unshredded_json_array(vec![
            Some(r#"{"a": "ok"}"#),
            None,
            Some(r#"{"a": null}"#),
            Some(r#"{"b": 2}"#),
        ])?;

        let result = execute_variant_get(arr, "$.a", None)?;

        assert_eq!(result.dtype(), &VortexDType::Variant(Nullability::Nullable));
        let mut ctx = SESSION.create_execution_ctx();
        let row0 = result.execute_scalar(0, &mut ctx)?;
        assert_eq!(
            row0.as_variant()
                .value()
                .and_then(|value| value.as_utf8().value())
                .map(|value| value.as_str()),
            Some("ok")
        );
        assert_nth_scalar_is_null!(result, 1, &mut ctx);
        assert_eq!(
            result
                .execute_scalar(2, &mut ctx)?
                .as_variant()
                .is_variant_null(),
            Some(true)
        );
        assert_nth_scalar_is_null!(result, 3, &mut ctx);

        Ok(())
    }

    fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(*value);
        }
        Arc::new(builder.finish())
    }

    fn nullable_binary_view_array(values: &[Option<&[u8]>]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            match value {
                Some(value) => builder.append_value(*value),
                None => builder.append_null(),
            }
        }
        Arc::new(builder.finish())
    }

    fn i32_variant_value(value: i32) -> Vec<u8> {
        let mut builder = VariantBuilder::new();
        builder.append_value(value);
        builder.finish().1
    }

    fn string_variant_value(value: &str) -> Vec<u8> {
        let mut builder = VariantBuilder::new();
        builder.append_value(value);
        builder.finish().1
    }

    fn object_with_b_value(value: &str) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        builder.new_object().with_field("b", value).finish();
        builder.finish()
    }

    fn object_with_a_and_b_value(a: i32, b: &str) -> (Vec<u8>, Vec<u8>) {
        let mut builder = VariantBuilder::new();
        builder
            .new_object()
            .with_field("a", a)
            .with_field("b", b)
            .finish();
        builder.finish()
    }

    fn make_partially_shredded_arrow_variant() -> VortexResult<ArrowVariantArray> {
        let (metadata0, root_value0) = object_with_a_and_b_value(99, "left");
        let (metadata1, root_value1) = object_with_b_value("right");
        let (metadata2, root_value2) = object_with_b_value("missing_a");

        let metadata = nullable_binary_view_array(&[
            Some(metadata0.as_slice()),
            Some(metadata1.as_slice()),
            Some(metadata2.as_slice()),
        ]);
        let value = nullable_binary_view_array(&[
            Some(root_value0.as_slice()),
            Some(root_value1.as_slice()),
            Some(root_value2.as_slice()),
        ]);

        let a_value1 = i32_variant_value(30);
        let a_value = nullable_binary_view_array(&[None, Some(a_value1.as_slice()), None]);
        let a_typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![Some(10), None, None]));
        let a_shredded: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, true)),
            ]
            .into(),
            vec![a_value, a_typed_value],
            None,
        )?);

        let b_value0 = string_variant_value("left");
        let b_value1 = string_variant_value("right");
        let b_value2 = string_variant_value("missing_a");
        let b_value = nullable_binary_view_array(&[
            Some(b_value0.as_slice()),
            Some(b_value1.as_slice()),
            Some(b_value2.as_slice()),
        ]);
        let b_shredded: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new("value", DataType::BinaryView, true))].into(),
            vec![b_value],
            None,
        )?);

        let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![
                Arc::new(Field::new("a", a_shredded.data_type().clone(), true)),
                Arc::new(Field::new("b", b_shredded.data_type().clone(), true)),
            ]
            .into(),
            vec![a_shredded, b_shredded],
            None,
        )?);

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_value.data_type().clone(),
                    true,
                )),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )?;
        Ok(ArrowVariantArray::try_new(&struct_array)?)
    }

    fn make_partially_shredded_parquet_array() -> VortexResult<ArrayRef> {
        let arrow_variant = make_partially_shredded_arrow_variant()?;
        let storage = arrow_variant.inner();
        let value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);
        let typed_value_nullable = storage
            .fields()
            .iter()
            .find(|field| field.name() == "typed_value")
            .map(|field| field.is_nullable())
            .unwrap_or(false);

        let metadata = SESSION
            .arrow()
            .from_arrow_array(ArrowArrayRef::clone(arrow_variant.metadata_column()), false)?;
        let value = Some(SESSION.arrow().from_arrow_array(
            ArrowArrayRef::clone(arrow_variant.value_column()),
            value_nullable,
        )?);
        let typed_value = arrow_variant
            .typed_value_column()
            .map(|typed_value| {
                SESSION
                    .arrow()
                    .from_arrow_array(ArrowArrayRef::clone(typed_value), typed_value_nullable)
            })
            .transpose()?;

        Ok(
            ParquetVariant::try_new(Validity::NonNullable, metadata, value, typed_value)?
                .into_array(),
        )
    }

    fn make_partially_shredded_object_array() -> VortexResult<ArrayRef> {
        let arrow_variant = make_partially_shredded_arrow_variant()?;
        let parquet_array = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = parquet_array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };
        Ok(canonical.into_array())
    }

    fn make_canonical_raw_core_with_shredded_child() -> VortexResult<ArrayRef> {
        let raw_core = make_unshredded_json_array(vec![
            Some(r#"{"a": 99, "b": "left"}"#),
            Some(r#"{"a": 30, "b": "right"}"#),
            Some(r#"{"b": "missing_a"}"#),
        ])?;
        let shredded = VortexStructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_option_iter([Some(10i32), None, None]).into_array(),
        )])?;

        Ok(VariantArray::try_new(raw_core, Some(shredded.into_array()))?.into_array())
    }

    fn make_canonical_raw_core_with_nested_nullable_shredded_child() -> VortexResult<ArrayRef> {
        let raw_core = make_unshredded_json_array(vec![
            Some(r#"{"a": {"b": 100}}"#),
            Some(r#"{"a": {"b": 200}}"#),
            Some(r#"{"a": {"b": 300}}"#),
        ])?;
        let nested = VortexStructArray::try_from_iter_with_validity(
            [("b", PrimitiveArray::from_iter([10i32, 20, 30]).into_array())],
            Validity::from_iter([false, true, true]),
        )?;
        let shredded = VortexStructArray::try_from_iter([("a", nested.into_array())])?.into_array();

        Ok(VariantArray::try_new(raw_core, Some(shredded))?.into_array())
    }

    fn assert_variant_i32_scalars(array: &ArrayRef, expected: &[Option<i32>]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = SESSION.create_execution_ctx();
        for (idx, expected) in expected.iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let variant = scalar.as_variant();
            match expected {
                Some(expected) => {
                    let value = variant
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null variant scalar"))?;
                    let value =
                        value.cast(&VortexDType::Primitive(PType::I32, Nullability::Nullable))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(*expected));
                }
                None => assert!(variant.is_null()),
            }
        }
        Ok(())
    }

    fn assert_variant_object_a_b(
        array: &ArrayRef,
        expected_a: &[Option<i32>],
        expected_b: &[&str],
    ) -> VortexResult<()> {
        assert_eq!(array.len(), expected_a.len());
        assert_eq!(array.len(), expected_b.len());
        let mut ctx = SESSION.create_execution_ctx();
        for idx in 0..array.len() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let object = scalar
                .as_variant()
                .value()
                .ok_or_else(|| vortex_err!("expected non-null variant object"))?
                .as_struct();

            match expected_a[idx] {
                Some(expected) => {
                    let value = object
                        .field("a")
                        .ok_or_else(|| vortex_err!("expected field a"))?
                        .as_variant()
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null field a"))?
                        .cast(&VortexDType::Primitive(PType::I32, Nullability::Nullable))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(expected));
                }
                None => assert!(object.field("a").is_none()),
            }

            let field_b = object
                .field("b")
                .ok_or_else(|| vortex_err!("expected field b"))?;
            let value = field_b
                .as_variant()
                .value()
                .ok_or_else(|| vortex_err!("expected non-null field b"))?;
            assert_eq!(
                value.as_utf8().value().map(|value| value.as_str()),
                Some(expected_b[idx])
            );
        }
        Ok(())
    }

    fn make_shredded_typed_array() -> VortexResult<ArrayRef> {
        let metadata = binary_view_array(&[b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![Some(10), None, Some(30)]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, true)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )?;
        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())
    }

    fn assert_typed_value_i32(
        array: &ArrayRef,
        expected: impl IntoIterator<Item = Option<i32>>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let executed = array
            .optimize_ctx(&SESSION)?
            .execute::<ArrayRef>(&mut ctx)?;
        let executed_variant = executed.as_::<ParquetVariant>();
        let typed_value = executed_variant
            .typed_value()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?;

        assert_arrays_eq!(
            typed_value,
            PrimitiveArray::from_option_iter(expected),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_slice_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;

        let sliced = arr.slice(1..3)?;
        assert_typed_value_i32(&sliced, [None, Some(30)])
    }

    #[test]
    fn test_filter_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let filtered = arr.filter(Mask::from_iter([true, true, false]))?;

        assert_typed_value_i32(&filtered, [Some(10), None])
    }

    #[test]
    fn test_take_shredded_typed_value() -> VortexResult<()> {
        let arr = make_shredded_typed_array()?;
        let taken = arr.take(PrimitiveArray::from_iter([2u64, 1, 0]).into_array())?;

        assert_typed_value_i32(&taken, [Some(30), None, Some(10)])
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_shredded_and_raw_fallback() -> VortexResult<()> {
        let parquet = make_partially_shredded_parquet_array()?;

        let result = execute_variant_get(
            parquet,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_direct_parquet_uses_raw_storage_for_unshredded_path() -> VortexResult<()> {
        let parquet = make_partially_shredded_parquet_array()?;

        let result = execute_variant_get(
            parquet,
            "$.b",
            Some(VortexDType::Utf8(Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            VarBinArray::from(vec![Some("left"), Some("right"), Some("missing_a")]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonicalized_parquet_uses_top_level_shredded() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;
        let canonical_variant = canonical.as_::<Variant>();

        let core_storage = canonical_variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(core_storage.typed_value().is_none());
        assert!(core_storage.value().is_some());

        let shredded = canonical_variant
            .shredded()
            .ok_or_else(|| vortex_err!("expected canonical shredded child"))?
            .clone()
            .execute::<VortexStructArray>(&mut SESSION.create_execution_ctx())?;
        assert_eq!(
            shredded.unmasked_field_by_name("a")?.dtype(),
            &VortexDType::Variant(Nullability::Nullable)
        );
        assert!(shredded.unmasked_field_by_name_opt("b").is_none());

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// Events shredded as `kind` and `commit.collection`, canonicalized. Only some events have a
    /// `commit`, and `commit.rev` stays in the residual value of `commit`.
    fn make_nested_shredded_events() -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(vec![
            r#"{"kind":"commit","commit":{"collection":"post","rev":1}}"#,
            r#"{"kind":"identity"}"#,
            r#"{"kind":"commit","commit":{"collection":"like","rev":2}}"#,
        ]));
        let shredding = ShreddedSchemaBuilder::new()
            .with_path("kind", &DataType::Utf8)?
            .with_path("commit.collection", &DataType::Utf8)?
            .build();
        let shredded = shred_variant(&json_to_variant(&json)?, &shredding)?;
        let parquet_array = ParquetVariant::from_arrow_variant(&shredded, &SESSION.arrow())?;
        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = parquet_array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };
        Ok(canonical.into_array())
    }

    /// The shapes in which a scan hands a Variant column to `variant_get`.
    #[derive(Clone, Copy, Debug)]
    enum Input {
        Canonical,
        Chunked,
        Filtered,
    }

    #[rstest]
    #[case::canonical(Input::Canonical, &[0, 1, 2])]
    #[case::chunked(Input::Chunked, &[0, 1, 2])]
    #[case::filtered(Input::Filtered, &[0, 2])]
    fn test_variant_get_nested_shredded_path(
        #[case] input: Input,
        #[case] rows: &[usize],
    ) -> VortexResult<()> {
        let events = make_nested_shredded_events()?;
        let array = match input {
            Input::Canonical => events,
            Input::Chunked => ChunkedArray::try_new(
                [events.slice(0..2)?, events.slice(2..3)?],
                events.dtype().clone(),
            )?
            .into_array(),
            Input::Filtered => events.filter(Mask::from_iter([true, false, true]))?,
        };
        let mut ctx = SESSION.create_execution_ctx();

        // `commit` is a partially shredded object, which holds the rest of the path.
        let collection = [Some("post"), None, Some("like")];
        assert_arrays_eq!(
            execute_variant_get(
                array.clone(),
                "$.commit.collection",
                Some(VortexDType::Utf8(Nullability::Nullable)),
            )?,
            VarBinViewArray::from_iter_nullable_str(rows.iter().map(|&row| collection[row])),
            &mut ctx
        );

        // A field that is not shredded comes from the residual value of `commit`.
        let rev = [Some(1i64), None, Some(2)];
        assert_arrays_eq!(
            execute_variant_get(
                array,
                "$.commit.rev",
                Some(VortexDType::Primitive(PType::I64, Nullability::Nullable)),
            )?,
            PrimitiveArray::from_option_iter(rows.iter().map(|&row| rev[row])),
            &mut ctx
        );
        Ok(())
    }

    /// Chunks of storage execute to one storage array, so that a chunked Variant compresses and
    /// deserializes as a single array.
    #[test]
    fn test_chunked_storage_executes_to_one_storage_array() -> VortexResult<()> {
        let first = make_unshredded_json_array(vec![Some(r#"{"a":1}"#), Some(r#"{"a":2}"#)])?;
        let second = make_unshredded_json_array(vec![Some(r#"{"a":3}"#)])?;
        let dtype = first.dtype().clone();
        let chunked = ChunkedArray::try_new([first, second], dtype)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let Canonical::Variant(canonical) = chunked.clone().execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };
        assert!(canonical.core_storage().is::<ParquetVariant>());

        let requested = VortexDType::Primitive(PType::I64, Nullability::Nullable);
        assert_arrays_eq!(
            execute_variant_get(chunked, "$.a", Some(requested))?,
            PrimitiveArray::from_option_iter([Some(1i64), Some(2), Some(3)]),
            &mut ctx
        );
        Ok(())
    }

    /// The struct that the Variant layout stores for [`make_nested_shredded_events`]: the core
    /// storage and the shredded child as fields, and an extraction of `path` as `dtype` from
    /// the Variant packed back from them, optimized.
    fn extraction_over_pack(
        path: &str,
        dtype: VortexDType,
        complete: bool,
    ) -> VortexResult<(ArrayRef, BoundExpression)> {
        let events = make_nested_shredded_events()?;
        let events = events.as_::<Variant>();
        let shredded = events
            .shredded()
            .ok_or_else(|| vortex_err!("expected a shredded child"))?
            .clone();
        let storage = VortexStructArray::try_new(
            FieldNames::from(["core", "shredded"]),
            vec![events.core_storage().clone(), shredded],
            events.len(),
            Validity::NonNullable,
        )?
        .into_array();

        let packed = variant_pack(
            get_item("core", root()),
            get_item("shredded", root()),
            complete,
        );
        let expr = variant_get(packed, parse_path(path)?, Some(dtype))
            .bind(storage.dtype())?
            .optimize_recursive()?;
        Ok((storage, expr))
    }

    /// An extraction from a packed Variant reads the shredded child alone when it holds the
    /// path, and the core storage alone when it does not.
    #[rstest]
    // A shredded leaf.
    #[case::shredded("$.kind", &[Some("commit"), Some("identity"), Some("commit")], "shredded")]
    // A leaf of a shredded object, which is a Variant of its own.
    #[case::nested("$.commit.collection", &[Some("post"), None, Some("like")], "shredded")]
    // A field that was not shredded is only in the core storage.
    #[case::unshredded("$.other", &[None, None, None], "core")]
    fn test_variant_get_over_complete_pack_reads_one_child(
        #[case] path: &str,
        #[case] expected: &[Option<&str>],
        #[case] child: &str,
    ) -> VortexResult<()> {
        let utf8 = VortexDType::Utf8(Nullability::Nullable);
        let (storage, expr) = extraction_over_pack(path, utf8, true)?;

        let read = expr.to_string();
        let other = if child == "core" { "shredded" } else { "core" };
        assert!(read.contains(child), "{read} must read {child}");
        assert!(!read.contains(other), "{read} must not read {other}");
        assert_arrays_eq!(
            storage.apply_bound(&expr)?,
            VarBinViewArray::from_iter_nullable_str(expected.iter().copied()),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// A shredded child that is not complete can have left a value in the core storage, so
    /// the extraction keeps both.
    #[test]
    fn test_variant_get_over_partial_pack_reads_both_children() -> VortexResult<()> {
        let utf8 = VortexDType::Utf8(Nullability::Nullable);
        let (storage, expr) = extraction_over_pack("$.kind", utf8, false)?;

        let read = expr.to_string();
        assert!(read.contains("core") && read.contains("shredded"), "{read}");
        assert_arrays_eq!(
            storage.apply_bound(&expr)?,
            VarBinViewArray::from_iter_nullable_str([
                Some("commit"),
                Some("identity"),
                Some("commit")
            ]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// Decoding the storage once leaves a storage array whose children need no more decoding,
    /// whether or not the Variant was filtered first, and extracts the same values.
    #[rstest]
    #[case::whole(false)]
    #[case::filtered(true)]
    fn test_decode_unshredded_storage(#[case] filtered: bool) -> VortexResult<()> {
        let rows = vec![Some(r#"{"a":1}"#), Some(r#"{"a":2}"#), Some(r#"{"a":3}"#)];
        let array = make_unshredded_json_array(rows)?;
        let (array, expected) = if filtered {
            (
                array.filter(Mask::from_iter([true, false, true]))?,
                [Some(1i64), Some(3)].as_slice(),
            )
        } else {
            (array, [Some(1i64), Some(2), Some(3)].as_slice())
        };
        let mut ctx = SESSION.create_execution_ctx();

        let decoded = decode_unshredded_storage(array, &mut ctx)?;

        let variant = decoded.as_::<Variant>();
        let storage = variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(storage.metadata().is_canonical());
        assert!(storage.value().is_some_and(|value| value.is_canonical()));
        assert_arrays_eq!(
            execute_variant_get(
                decoded,
                "$.a",
                Some(VortexDType::Primitive(PType::I64, Nullability::Nullable)),
            )?,
            PrimitiveArray::from_option_iter(expected.iter().copied()),
            &mut ctx
        );
        Ok(())
    }

    /// A shredded Variant is left alone: its extractions read the shredded children.
    #[test]
    fn test_decode_unshredded_storage_keeps_shredded() -> VortexResult<()> {
        let events = make_nested_shredded_events()?;
        let mut ctx = SESSION.create_execution_ctx();

        let decoded = decode_unshredded_storage(events.clone(), &mut ctx)?;

        assert!(decoded.as_::<Variant>().shredded().is_some());
        assert_arrays_eq!(
            execute_variant_get(
                decoded,
                "$.kind",
                Some(VortexDType::Utf8(Nullability::Nullable)),
            )?,
            execute_variant_get(
                events,
                "$.kind",
                Some(VortexDType::Utf8(Nullability::Nullable)),
            )?,
            &mut ctx
        );
        Ok(())
    }

    /// Rows whose value does not fit the shredded type stay in the residual value, and are
    /// extracted from there into the positions the typed column leaves null.
    #[rstest]
    #[case::as_string(DataType::Utf8, VortexDType::Utf8(Nullability::Nullable))]
    #[case::as_integer(
        DataType::Int64,
        VortexDType::Primitive(PType::I64, Nullability::Nullable)
    )]
    fn test_variant_get_fills_rows_the_shredded_column_lacks(
        #[case] shredded_type: DataType,
        #[case] requested: VortexDType,
    ) -> VortexResult<()> {
        let rows = vec![
            Some(r#"{"a":"x"}"#),
            Some(r#"{"a":7}"#),
            Some(r#"{"b":1}"#),
            Some(r#"{"a":"y"}"#),
            Some(r#"{"a":8}"#),
            Some(r#"{"a":true}"#),
        ];
        let unshredded = make_unshredded_json_array(rows.clone())?;

        let json: ArrowArrayRef = Arc::new(StringArray::from(rows));
        let shredding = ShreddedSchemaBuilder::new()
            .with_path("a", &shredded_type)?
            .build();
        let shredded = shred_variant(&json_to_variant(&json)?, &shredding)?;
        let shredded = ParquetVariant::from_arrow_variant(&shredded, &SESSION.arrow())?;
        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(shredded) = shredded.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };

        // Shredding must not change what a path extracts to.
        let expected = execute_variant_get(unshredded, "$.a", Some(requested.clone()))?;
        assert!(
            !expected
                .validity()?
                .execute_mask(expected.len(), &mut ctx)?
                .all_false(),
            "the fixture must extract some values"
        );
        assert_arrays_eq!(
            execute_variant_get(shredded.into_array(), "$.a", Some(requested))?,
            expected,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_child_overrides_raw_core() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10), Some(30), None]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_respects_nested_validity() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(
            canonical,
            "$.a.b",
            Some(VortexDType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(100), Some(20), Some(30)]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_variant_get_canonical_shredded_untyped_uses_typed_rows() -> VortexResult<()> {
        let canonical = make_canonical_raw_core_with_nested_nullable_shredded_child()?;

        let result = execute_variant_get(canonical, "$.a.b", None)?;

        assert_variant_i32_scalars(&result, &[Some(100), Some(20), Some(30)])
    }

    #[test]
    fn test_variant_get_untyped_partial_object_preserves_raw_fields() -> VortexResult<()> {
        let canonical = make_partially_shredded_object_array()?;

        let direct_result =
            execute_variant_get(make_partially_shredded_parquet_array()?, "$", None)?;
        assert_variant_object_a_b(
            &direct_result,
            &[Some(10), Some(30), None],
            &["left", "right", "missing_a"],
        )?;

        let canonical_result = execute_variant_get(canonical, "$", None)?;
        assert_variant_object_a_b(
            &canonical_result,
            &[Some(10), Some(30), None],
            &["left", "right", "missing_a"],
        )
    }

    #[test]
    fn test_canonicalization_moves_shredded_child_out_of_core_storage() -> VortexResult<()> {
        let parquet_array = make_partially_shredded_parquet_array()?;
        assert!(
            parquet_array
                .as_::<ParquetVariant>()
                .typed_value()
                .is_some()
        );

        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = parquet_array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };

        assert!(canonical.shredded().is_some());
        let core_storage = canonical
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        assert!(core_storage.typed_value().is_none());
        assert!(core_storage.value().is_some());

        Ok(())
    }
}
