// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JSON extension wrappers for Parquet Variant storage.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyArrayData;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_json::Json;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

mod variant_to_json_children {
    pub const VARIANT: usize = 0;
    pub const NUM_SLOTS: usize = 1;
    pub const SLOT_NAMES: [&str; NUM_SLOTS] = ["variant"];
}

/// Array that exposes a Variant array as JSON strings.
#[derive(Debug, Clone)]
pub struct VariantToJson;

/// A [`VariantToJson`]-encoded array.
pub type VariantToJsonArray = Array<VariantToJson>;

impl VariantToJson {
    /// Creates a JSON wrapper around a Variant-typed array.
    pub fn try_new(variant: ArrayRef) -> VortexResult<VariantToJsonArray> {
        vortex_ensure!(
            variant.dtype().is_variant(),
            "VariantToJson expects a Variant array, got {}",
            variant.dtype()
        );

        let storage_dtype = DType::Utf8(variant.dtype().nullability());
        let dtype =
            DType::Extension(ExtDType::<Json>::try_new(EmptyMetadata, storage_dtype)?.erased());
        let len = variant.len();

        Array::try_from_parts(
            ArrayParts::new(VariantToJson, dtype, len, EmptyArrayData)
                .with_slots(vec![Some(variant)].into()),
        )
    }
}

impl VTable for VariantToJson {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant_to_json");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == variant_to_json_children::NUM_SLOTS,
            "VariantToJsonArray expects {} slots, got {}",
            variant_to_json_children::NUM_SLOTS,
            slots.len()
        );
        let variant = slots[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?;

        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("VariantToJsonArray dtype must be a JSON extension, got {dtype}");
        };
        vortex_ensure!(
            ext_dtype.is::<Json>(),
            "VariantToJsonArray dtype must be a JSON extension, got {dtype}"
        );
        vortex_ensure!(
            variant.dtype() == &DType::Variant(dtype.nullability()),
            "VariantToJsonArray child dtype {} does not match JSON dtype nullability {}",
            variant.dtype(),
            dtype
        );
        vortex_ensure!(
            variant.len() == len,
            "VariantToJsonArray child length {} does not match outer length {}",
            variant.len(),
            len
        );

        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantToJsonArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            metadata.is_empty(),
            "VariantToJsonArray metadata must be empty"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "VariantToJsonArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == variant_to_json_children::NUM_SLOTS,
            "VariantToJsonArray expects {} children, got {}",
            variant_to_json_children::NUM_SLOTS,
            children.len()
        );

        let variant_dtype = DType::Variant(dtype.nullability());
        let variant = children.get(variant_to_json_children::VARIANT, &variant_dtype, len)?;

        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(vec![Some(variant)].into()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match variant_to_json_children::SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantToJsonArray slot_name index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let variant = array.as_ref().slots()[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?;
        let variant = variant.clone().execute::<VariantArray>(ctx)?;
        vortex_ensure!(
            variant.shredded().is_none(),
            "VariantToJsonArray can only export unshredded Parquet Variant storage to JSON"
        );

        let parquet_variant = variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| {
                vortex_err!(
                    "VariantToJsonArray requires Parquet Variant core storage, got {}",
                    variant.core_storage().encoding_id()
                )
            })?;
        vortex_ensure!(
            parquet_variant.typed_value_array().is_none(),
            "VariantToJsonArray can only export unshredded Parquet Variant storage to JSON"
        );
        let value = parquet_variant.value_array().ok_or_else(|| {
            vortex_err!("VariantToJsonArray requires Parquet Variant value storage")
        })?;
        let arrow_variant = crate::arrow::export_storage_to_target(
            &parquet_variant,
            &vec![
                Arc::new(Field::new("metadata", DataType::Binary, false)),
                Arc::new(Field::new(
                    "value",
                    DataType::Binary,
                    value.dtype().is_nullable(),
                )),
            ]
            .into(),
            ctx,
        )?;
        let arrow_json = parquet_variant_compute::variant_to_json(&arrow_variant)?;
        let storage = ArrayRef::from_arrow(&arrow_json, array.dtype().is_nullable())?;

        Ok(ExecutionResult::done(
            ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage)?.into_array(),
        ))
    }
}

impl ValidityVTable<VariantToJson> for VariantToJson {
    fn validity(array: ArrayView<'_, VariantToJson>) -> VortexResult<Validity> {
        array.slots()[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?
            .validity()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::session::ArraySession;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    });

    #[test]
    fn variant_to_json_canonicalizes_to_json_extension() -> VortexResult<()> {
        let values = [
            "0".to_string(),
            r#"{"a":32}"#.to_string(),
            r#""hello""#.to_string(),
            "null".to_string(),
        ];
        let storage =
            VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();
        let source =
            ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage.clone())?.into_array();

        let mut exec_ctx = SESSION.create_execution_ctx();
        let arrow_array = {
            let session = exec_ctx.session().clone();
            session
                .arrow()
                .execute_arrow(storage, None, &mut exec_ctx)?
        };
        let arrow_variant = parquet_variant_compute::json_to_variant(&arrow_array)?;
        let variant = ParquetVariant::from_arrow_variant(&arrow_variant)?;

        let wrapped = VariantToJson::try_new(variant)?;
        assert_eq!(wrapped.dtype(), source.dtype());

        let json = wrapped
            .into_array()
            .execute::<ExtensionArray>(&mut exec_ctx)?;
        assert_eq!(json.dtype(), source.dtype());
        assert!(json.storage_array().dtype().is_utf8());
        let json_storage = json
            .storage_array()
            .clone()
            .execute::<VarBinViewArray>(&mut exec_ctx)?;
        let actual = json_storage.with_iterator(|iter| {
            iter.map(|value| value.map(<[u8]>::to_vec))
                .collect::<Vec<_>>()
        });
        let expected = values
            .iter()
            .map(|value| Some(value.as_bytes().to_vec()))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);

        Ok(())
    }
}
