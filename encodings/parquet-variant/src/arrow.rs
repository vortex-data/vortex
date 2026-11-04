// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantBuilder;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::unshred_variant;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Struct;
use vortex_array::arrays::Variant;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::masked::MaskedArrayExt;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::matcher::Matcher;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::to_arrow_null_buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;
use crate::array::parquet_typed_value_from_logical_shredded;

/// Arrow canonical extension name for Parquet Variant storage.
const PARQUET_VARIANT_ARROW_EXTENSION_NAME: &str = "arrow.parquet.variant";
static ARROW_PARQUET_VARIANT: CachedId = CachedId::new(PARQUET_VARIANT_ARROW_EXTENSION_NAME);

fn parquet_variant_storage_request(fields: &Fields) -> Option<(bool, bool)> {
    let mut has_metadata = false;
    let mut has_value = false;
    let mut has_typed_value = false;

    for field in fields {
        match field.name().as_str() {
            "metadata" if !has_metadata => has_metadata = true,
            "value" if !has_value => has_value = true,
            "typed_value" if !has_typed_value => has_typed_value = true,
            _ => return None,
        }
    }

    (has_metadata && (has_value || has_typed_value)).then_some((has_value, has_typed_value))
}

pub(crate) fn export_storage_to_target<T: ParquetVariantArrayExt>(
    parquet_array: &T,
    target_fields: &Fields,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let session = ctx.session().clone();
    let arrow = session.arrow();
    let exporter = arrow.exporter(options);
    let mut arrays = Vec::with_capacity(target_fields.len());

    for field in target_fields {
        let child = match field.name().as_str() {
            "metadata" => Some(parquet_array.metadata().clone()),
            "value" => parquet_array.value().cloned(),
            "typed_value" => parquet_array.typed_value().cloned(),
            _ => unreachable!("storage fields were validated before export"),
        };
        let Some(child) = child else {
            vortex_bail!(
                "cannot export Parquet Variant storage field '{}' because source has no {} child",
                field.name(),
                field.name()
            );
        };

        arrays.push(exporter.execute_arrow(child, Some(field.as_ref()), ctx)?);
    }

    let nulls = to_arrow_null_buffer(
        ParquetVariantArrayExt::parquet_variant_validity(parquet_array),
        parquet_array.as_ref().len(),
        ctx,
    )?;
    Ok(Arc::new(StructArray::try_new(
        target_fields.clone(),
        arrays,
        nulls,
    )?))
}

pub(crate) fn export_unshredded_storage_to_target<T: ParquetVariantArrayExt>(
    parquet_array: &T,
    target_fields: &Fields,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let arrow_variant = parquet_array.to_arrow(options, ctx)?;
    let unshredded = unshred_variant(&arrow_variant)?;
    let unshredded_array = if parquet_array.as_ref().dtype().is_nullable() {
        ParquetVariant::from_arrow_variant_nullable(&unshredded, &ctx.session().arrow())?
    } else {
        ParquetVariant::from_arrow_variant(&unshredded, &ctx.session().arrow())?
    };
    let unshredded_parquet = unshredded_array.as_::<ParquetVariant>();
    export_storage_to_target(&unshredded_parquet, target_fields, options, ctx)
}

/// Stop before canonicalization hides constant and masked storage behind a Variant wrapper.
struct ParquetVariantExportable;

impl Matcher for ParquetVariantExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<ParquetVariant>() || array.is::<Constant>() || array.is::<Masked>())
            .then_some(array)
    }
}

fn prepare_variant_for_export(
    array: ArrayRef,
    typed_value: Option<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = array.execute_until::<ParquetVariantExportable>(ctx)?;

    if let Some(variant) = array.as_opt::<Variant>() {
        let typed_value = match variant.shredded() {
            Some(shredded) => {
                vortex_ensure!(
                    typed_value.is_none(),
                    "multiple levels of Variant shredding are unsupported during Arrow export"
                );
                Some(parquet_typed_value_from_logical_shredded(
                    shredded.clone(),
                    ctx,
                )?)
            }
            None => typed_value,
        };

        return prepare_variant_for_export(variant.core_storage().clone(), typed_value, ctx);
    }

    if array.is::<ParquetVariant>() {
        return attach_typed_value(array, typed_value);
    }

    // `MaskedArray` children have no nulls, so the mask alone is the validity of the result.
    if let Some(masked) = array.as_opt::<Masked>() {
        let child = prepare_variant_for_export(masked.child().clone(), typed_value, ctx)?;
        let storage = child.as_::<ParquetVariant>();

        return ParquetVariant::try_new(
            masked.masked_validity(),
            storage.metadata().clone(),
            storage.value().cloned(),
            storage.typed_value().cloned(),
        )
        .map(IntoArray::into_array);
    }

    if let Some(constant) = array.as_opt::<Constant>() {
        let scalar = constant.scalar();
        if scalar.is_null() || scalar.as_variant().is_variant_null() == Some(true) {
            return constant_null_variant_storage(scalar, array.len(), typed_value);
        }

        vortex_bail!("cannot export non-null constant Variant to Arrow");
    }

    vortex_bail!("cannot export Variant without ParquetVariant core storage")
}

fn attach_typed_value(storage: ArrayRef, typed_value: Option<ArrayRef>) -> VortexResult<ArrayRef> {
    let Some(typed_value) = typed_value else {
        return Ok(storage);
    };

    let storage = storage.as_::<ParquetVariant>();
    vortex_ensure!(
        storage.typed_value().is_none(),
        "multiple levels of Variant shredding are unsupported during Arrow export"
    );

    ParquetVariant::try_new(
        ParquetVariantArrayExt::parquet_variant_validity(&storage),
        storage.metadata().clone(),
        storage.value().cloned(),
        Some(typed_value),
    )
    .map(IntoArray::into_array)
}

fn constant_null_variant_storage(
    scalar: &Scalar,
    len: usize,
    typed_value: Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    // All rows of an outer-null constant are null, so no row reads shredded field names.
    let mut field_names = Vec::new();
    if !scalar.is_null()
        && let Some(typed_value) = &typed_value
    {
        collect_shredded_field_names(typed_value, &mut field_names)?;
    }

    let mut builder =
        VariantBuilder::new().with_field_names(field_names.iter().map(|s| s.as_ref()));
    builder.append_value(PqVariant::Null);
    let (metadata, value) = builder.finish();

    let validity = if scalar.is_null() {
        Validity::AllInvalid
    } else if scalar.dtype().is_nullable() {
        Validity::AllValid
    } else {
        Validity::NonNullable
    };

    let value =
        ConstantArray::new(Scalar::binary(value, Nullability::NonNullable), len).into_array();
    // Synthesized nulls are fallbacks, not residual objects to merge with shredded fields.
    let value = match &typed_value {
        Some(typed_value) => value.mask(typed_value.is_null()?)?,
        None => value,
    };

    ParquetVariant::try_new(
        validity,
        ConstantArray::new(Scalar::binary(metadata, Nullability::NonNullable), len).into_array(),
        Some(value),
        typed_value,
    )
    .map(IntoArray::into_array)
}

// Walk the encodings, not the dtype: `parquet_typed_value_from_logical_shredded` adds field
// wrappers only under `List` and `Struct` encodings, and passes other encodings through as
// leaves. Only object names belong in metadata; residual field bytes require their original
// metadata dictionary.
fn collect_shredded_field_names(
    typed_value: &ArrayRef,
    names: &mut Vec<FieldName>,
) -> VortexResult<()> {
    if let Some(list) = typed_value.as_opt::<List>() {
        return collect_shredded_wrapper_names(list.elements(), names);
    }

    if let Some(object) = typed_value.as_opt::<Struct>() {
        for (name, field) in object.names().iter().zip(object.iter_unmasked_fields()) {
            names.push(name.clone());
            collect_shredded_wrapper_names(field, names)?;
        }
    }

    Ok(())
}

fn collect_shredded_wrapper_names(
    wrapper: &ArrayRef,
    names: &mut Vec<FieldName>,
) -> VortexResult<()> {
    let Some(wrapper) = wrapper.as_opt::<Struct>() else {
        vortex_bail!("expected a Parquet shredded field wrapper");
    };

    vortex_ensure!(
        wrapper.unmasked_field_by_name_opt("value").is_none(),
        "nested residual values over a null-constant Variant core are unsupported during Arrow export"
    );

    collect_shredded_field_names(wrapper.unmasked_field_by_name("typed_value")?, names)
}

impl ArrowExportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn vortex_id(&self) -> Id {
        ParquetVariant.id()
    }

    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        // Variant field inference is handled by `ArrowSession::to_arrow_field` directly; this
        // plugin hook is only consulted for `DType::Extension`, never for `DType::Variant`.
        Ok(None)
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        if target
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
            || !array.dtype().is_variant()
        {
            return Ok(ArrowExport::Unsupported(array));
        }

        let parquet_array = prepare_variant_for_export(array, None, ctx)?;
        let parquet_array = parquet_array.as_::<ParquetVariant>();

        if let DataType::Struct(fields) = target.data_type()
            && let Some((request_has_value, request_has_typed_value)) =
                parquet_variant_storage_request(fields)
        {
            let has_value = parquet_array.value().is_some();
            let has_typed_value = parquet_array.typed_value().is_some();

            if request_has_value && !request_has_typed_value && has_typed_value {
                return Ok(ArrowExport::Exported(export_unshredded_storage_to_target(
                    &parquet_array,
                    fields,
                    options,
                    ctx,
                )?));
            }
            if has_value && !request_has_value {
                vortex_bail!("cannot export Parquet Variant storage without losing value child");
            }
            if has_typed_value && !request_has_typed_value {
                vortex_bail!(
                    "cannot export Parquet Variant storage without losing typed_value child"
                );
            }

            return Ok(ArrowExport::Exported(export_storage_to_target(
                &parquet_array,
                fields,
                options,
                ctx,
            )?));
        }

        let arrow_variant =
            Arc::new(parquet_array.to_arrow(options, ctx)?.into_inner()) as ArrowArrayRef;

        if arrow_variant.data_type() == target.data_type() {
            Ok(ArrowExport::Exported(arrow_variant))
        } else {
            Ok(ArrowExport::Exported(parquet_variant_compute::variant_get(
                &arrow_variant,
                GetOptions::new().with_as_type(Some(target.clone().into())),
            )?))
        }
    }
}

impl ArrowImportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        if field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
        {
            return Ok(None);
        }

        Ok(Some(DType::Variant(field.is_nullable().into())))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        if !dtype.is_variant()
            || field
                .metadata()
                .get(EXTENSION_TYPE_NAME_KEY)
                .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
            || !matches!(array.data_type(), DataType::Struct(_))
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let arrow_variant = ArrowVariantArray::try_new(array.as_struct())?;
        let imported = if dtype.is_nullable() {
            ParquetVariant::from_arrow_variant_nullable(&arrow_variant, session)?
        } else {
            ParquetVariant::from_arrow_variant(&arrow_variant, session)?
        };
        Ok(ArrowImport::Imported(imported.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StructArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray as VortexStructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use super::PARQUET_VARIANT_ARROW_EXTENSION_NAME;
    use crate::ParquetVariant;

    #[fixture]
    fn session() -> VortexSession {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    }

    fn export_variant(session: &VortexSession, array: ArrayRef) -> VortexResult<ArrowVariantArray> {
        let mut ctx = session.create_execution_ctx();
        let field = session.arrow().to_arrow_field("variant", array.dtype())?;
        let result = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;

        Ok(ArrowVariantArray::try_new(result.as_ref())?)
    }

    fn variant_null_core(len: usize) -> ArrayRef {
        ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), len).into_array()
    }

    /// Shredded object `{a}` where `a` is a partially shredded Variant with residual bytes.
    fn shredded_with_nested_residual() -> VortexResult<ArrayRef> {
        let mut builder = VariantBuilder::new();
        let mut obj = builder.new_object();
        obj.insert("x", 1i32);
        obj.finish();
        let (meta, val) = builder.finish();

        let nested_core = ParquetVariant::try_new(
            Validity::NonNullable,
            VarBinViewArray::from_iter_bin([meta.as_slice(), meta.as_slice()]).into_array(),
            Some(VarBinViewArray::from_iter_bin([val.as_slice(), val.as_slice()]).into_array()),
            None,
        )?
        .into_array();
        let nested = VariantArray::try_new(nested_core, None)?.into_array();

        Ok(VortexStructArray::try_new(
            FieldNames::from_iter(["a"]),
            vec![nested],
            2,
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[rstest]
    #[case::outer_null(Scalar::null(DType::Variant(Nullability::Nullable)), true)]
    #[case::variant_null(Scalar::variant(Scalar::null(DType::Null)), false)]
    #[case::nullable_variant_null(Scalar::variant(Scalar::null(DType::Null)).into_nullable(), false)]
    fn test_export_constant_null_variant(
        session: VortexSession,
        #[case] scalar: Scalar,
        #[case] is_null: bool,
        #[values(0, 3)] len: usize,
        #[values(false, true)] wrap_canonical: bool,
    ) -> VortexResult<()> {
        let array = ConstantArray::new(scalar, len).into_array();
        let array = if wrap_canonical {
            VariantArray::try_new(array, None)?.into_array()
        } else {
            array
        };

        let result = export_variant(&session, array)?;

        assert_eq!(result.len(), len);
        for index in 0..len {
            assert_eq!(result.is_null(index), is_null);
            if !is_null {
                assert_eq!(result.value(index), PqVariant::Null);
            }
        }
        Ok(())
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum MaskAt {
        Nothing,
        Core,
        Variant,
    }

    #[rstest]
    fn test_export_shredded_values_over_null_core(
        session: VortexSession,
        #[values(MaskAt::Nothing, MaskAt::Core, MaskAt::Variant)] mask_at: MaskAt,
    ) -> VortexResult<()> {
        let mut ctx = session.create_execution_ctx();
        let validity = Validity::from(BitBuffer::from_iter([true, true, false]));

        let core = if mask_at == MaskAt::Core {
            MaskedArray::try_new(variant_null_core(3), validity.clone(), &mut ctx)?.into_array()
        } else {
            variant_null_core(3)
        };
        let shredded = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
        let array = VariantArray::try_new(core, Some(shredded))?.into_array();
        let array = if mask_at == MaskAt::Variant {
            MaskedArray::try_new(array, validity, &mut ctx)?.into_array()
        } else {
            array
        };

        let result = export_variant(&session, array)?;

        let masked = mask_at != MaskAt::Nothing;
        assert_eq!(result.value(0), PqVariant::Int32(10));
        assert_eq!(result.value(1), PqVariant::Null);
        assert_eq!(result.is_null(2), masked);
        if !masked {
            assert_eq!(result.value(2), PqVariant::Int32(30));
        }
        assert_eq!(result.inner().null_count(), usize::from(masked));
        Ok(())
    }

    #[rstest]
    fn test_export_object_shredding_over_null_core(session: VortexSession) -> VortexResult<()> {
        let mut ctx = session.create_execution_ctx();
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();

        // Object names that match Parquet wrapper names must still go into the metadata.
        let elements = VortexStructArray::try_new(
            FieldNames::from_iter(["typed_value"]),
            vec![PrimitiveArray::from_iter([42i32, 7]).into_array()],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter([0u32, 1, 1, 2]).into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();

        let shredded = VortexStructArray::try_new(
            FieldNames::from_iter(["a", "value"]),
            vec![a, list],
            3,
            Validity::from(BitBuffer::from_iter([true, true, false])),
        )?
        .into_array();
        let array = VariantArray::try_new(variant_null_core(3), Some(shredded))?.into_array();

        let result = export_variant(&session, array)?;

        assert_eq!(result.len(), 3);

        let first = result.try_value(0)?;
        assert_eq!(first.get_object_field("a"), Some(PqVariant::Int32(1)));
        let item = first
            .get_object_field("value")
            .and_then(|list| list.get_list_element(0))
            .and_then(|item| item.get_object_field("typed_value"));
        assert_eq!(item, Some(PqVariant::Int32(42)));

        let missing_field = result.try_value(1)?;
        assert!(matches!(missing_field, PqVariant::Object(_)));
        assert_eq!(missing_field.get_object_field("a"), None);

        assert_eq!(result.try_value(2)?, PqVariant::Null);
        Ok(())
    }

    #[rstest]
    fn test_export_rejects_nested_residual_over_null_core(
        session: VortexSession,
    ) -> VortexResult<()> {
        let shredded = shredded_with_nested_residual()?;
        let array = VariantArray::try_new(variant_null_core(2), Some(shredded))?.into_array();

        let Err(error) = export_variant(&session, array) else {
            vortex_bail!("export must reject a nested residual over a variant-null core");
        };

        assert!(
            error.to_string().contains(
                "nested residual values over a null-constant Variant core are unsupported"
            ),
            "{error}"
        );
        Ok(())
    }

    #[rstest]
    fn test_export_nested_residual_over_outer_null_core(
        session: VortexSession,
    ) -> VortexResult<()> {
        let core =
            ConstantArray::new(Scalar::null(DType::Variant(Nullability::Nullable)), 2).into_array();
        let shredded = shredded_with_nested_residual()?;
        let array = VariantArray::try_new(core, Some(shredded))?.into_array();

        let result = export_variant(&session, array)?;

        assert_eq!(result.len(), 2);
        assert_eq!(result.inner().null_count(), 2);
        Ok(())
    }

    #[rstest]
    fn test_export_rejects_nested_shredding_under_mask(session: VortexSession) -> VortexResult<()> {
        let mut ctx = session.create_execution_ctx();
        let (meta, val) = VariantBuilder::new().with_value("residual").finish();
        let inner_core = ParquetVariant::try_new(
            Validity::NonNullable,
            VarBinViewArray::from_iter_bin([meta.as_slice(); 3]).into_array(),
            Some(
                VarBinViewArray::from_iter_nullable_bin([None, None, Some(val.as_slice())])
                    .into_array(),
            ),
            None,
        )?
        .into_array();
        let inner_shredded =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None]).into_array();
        let inner = VariantArray::try_new(inner_core, Some(inner_shredded))?.into_array();

        let masked = MaskedArray::try_new(
            inner,
            Validity::from(BitBuffer::from_iter([true, true, true])),
            &mut ctx,
        )?
        .into_array();
        let outer_shredded =
            PrimitiveArray::from_option_iter([Some(100i32), None, None]).into_array();
        let array = VariantArray::try_new(masked, Some(outer_shredded))?.into_array();

        let Err(error) = export_variant(&session, array) else {
            vortex_bail!("export must reject two levels of Variant shredding");
        };

        assert!(
            error
                .to_string()
                .contains("multiple levels of Variant shredding are unsupported"),
            "{error}"
        );
        Ok(())
    }

    fn arrow_variant_storage() -> StructArray {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i8));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from("vortex"));
        builder.build().into_inner()
    }

    fn arrow_variant_field(storage: &StructArray) -> Field {
        Field::new("variant", storage.data_type().clone(), false).with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        )
    }

    fn assert_struct_arrays_eq(actual: &StructArray, expected: &StructArray) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.column_names(), expected.column_names());
        assert_eq!(actual.fields(), expected.fields());
        assert_eq!(actual.nulls(), expected.nulls());
        for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
            assert_eq!(actual.to_data(), expected.to_data());
        }
    }

    fn assert_variant_scalars_eq(
        actual: &ArrayRef,
        expected: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<()> {
        assert_eq!(actual.len(), expected.len());
        let mut actual_ctx = session.create_execution_ctx();
        let mut expected_ctx = session.create_execution_ctx();
        for index in 0..actual.len() {
            assert_eq!(
                actual.execute_scalar(index, &mut actual_ctx)?,
                expected.execute_scalar(index, &mut expected_ctx)?
            );
        }
        Ok(())
    }

    #[rstest]
    fn roundtrip_parquet_variant_extension_array_from_arrow(
        session: VortexSession,
    ) -> VortexResult<()> {
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let imported = session
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()) as ArrowArrayRef, &field)?;

        assert_eq!(imported.dtype(), &DType::Variant(Nullability::NonNullable));
        assert!(imported.as_opt::<ParquetVariant>().is_some());

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(imported, Some(&field), &mut ctx)?;
        let exported = exported.as_struct();

        assert_struct_arrays_eq(exported, &storage);
        Ok(())
    }

    #[rstest]
    fn export_canonical_variant_with_parquet_variant_core_storage(
        session: VortexSession,
    ) -> VortexResult<()> {
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let core_storage = session
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()) as ArrowArrayRef, &field)?;
        let canonical = VariantArray::try_new(core_storage, None)?.into_array();

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(canonical, Some(&field), &mut ctx)?;
        let exported = exported.as_struct();

        assert_struct_arrays_eq(exported, &storage);
        Ok(())
    }

    #[rstest]
    fn export_canonical_variant_reattaches_shredded_child(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(10i32).finish(),
            VariantBuilder::new().with_value(20i32).finish(),
            VariantBuilder::new().with_value(30i32).finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();
        let expected =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();

        let mut ctx = session.create_execution_ctx();
        let canonical = expected
            .clone()
            .execute::<VariantArray>(&mut ctx)?
            .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, false),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let exported = session
            .arrow()
            .execute_arrow(canonical, Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());

        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
    }

    #[rstest]
    fn roundtrip_parquet_variant_extension_array_from_vortex(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(42i32).finish(),
            VariantBuilder::new().with_value(true).finish(),
            VariantBuilder::new().with_value("vortex").finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();
        let value = VarBinViewArray::from_iter_bin(rows.iter().map(|(_, value)| value.as_slice()))
            .into_array();
        let array = ParquetVariant::try_new(Validity::NonNullable, metadata, Some(value), None)?
            .into_array();
        let expected = array.clone();

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array.clone(), None, &mut ctx)?;

        let field = Field::new(
            "",
            exported.data_type().clone(),
            array.dtype().is_nullable(),
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let actual = session
            .arrow()
            .from_arrow_array(Arc::clone(&exported), &field)?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    fn export_fully_shredded_to_inferred_storage_unshreds_value(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(10i32).finish(),
            VariantBuilder::new().with_value(20i32).finish(),
            VariantBuilder::new().with_value(30i32).finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();
        let typed_value = buffer![10i32, 20, 30].into_array();
        let expected =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(expected.clone(), None, &mut ctx)?;

        assert_eq!(
            exported.data_type(),
            &DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, false),
                ]
                .into()
            )
        );

        let field = Field::new("", exported.data_type().clone(), false).with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );
        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
    }

    #[rstest]
    fn export_partially_shredded_to_metadata_value_preserves_typed_rows(
        session: VortexSession,
    ) -> VortexResult<()> {
        let (metadata0, value0) = VariantBuilder::new().with_value("fallback-0").finish();
        let (metadata1, _value1) = VariantBuilder::new().with_value(20i32).finish();
        let (metadata2, value2) = VariantBuilder::new().with_value("fallback-2").finish();
        let metadata = VarBinViewArray::from_iter_bin([
            metadata0.as_slice(),
            metadata1.as_slice(),
            metadata2.as_slice(),
        ])
        .into_array();
        let value = VarBinViewArray::from_iter_nullable_bin([
            Some(value0.as_slice()),
            None,
            Some(value2.as_slice()),
        ])
        .into_array();
        let typed_value = PrimitiveArray::from_option_iter([None, Some(20i32), None]).into_array();
        let expected = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?
        .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(expected.clone(), Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());

        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
    }

    #[rstest]
    fn export_partial_storage_target_rejects_data_loss(session: VortexSession) -> VortexResult<()> {
        let (metadata0, value0) = VariantBuilder::new().with_value("fallback-0").finish();
        let (metadata1, _value1) = VariantBuilder::new().with_value(20i32).finish();
        let metadata = VarBinViewArray::from_iter_bin([metadata0.as_slice(), metadata1.as_slice()])
            .into_array();
        let value =
            VarBinViewArray::from_iter_nullable_bin([Some(value0.as_slice()), None]).into_array();
        let typed_value = PrimitiveArray::from_option_iter([None, Some(20i32)]).into_array();
        let array = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?
        .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("typed_value", DataType::Int32, true),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let mut ctx = session.create_execution_ctx();
        let err = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("cannot export Parquet Variant storage without losing value child"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[rstest]
    fn roundtrip_shredded_parquet_variant_extension_array_from_vortex(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(10i32).finish(),
            VariantBuilder::new().with_value(20i32).finish(),
            VariantBuilder::new().with_value(30i32).finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();

        let typed_value = buffer![10i32, 20, 30].into_array();
        let array =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();
        let expected = array.clone();

        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::Binary, false),
                    Field::new("typed_value", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );
        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;
        assert_eq!(
            exported.data_type(),
            field.data_type(),
            "Parquet Variant export should honor the requested storage schema"
        );

        let actual = session.arrow().from_arrow_array(exported, &field)?;

        assert_variant_scalars_eq(&actual, &expected, &session)
    }
}
