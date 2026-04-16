// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ParquetVariant> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ParquetVariant)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ParquetVariant)),
]);

impl SliceKernel for ParquetVariant {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?.slice(range.clone())?;
        let metadata = array.metadata_array().slice(range.clone())?;
        let value = array
            .value_array()
            .map(|v| v.slice(range.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
            .map(|tv| tv.slice(range))
            .transpose()?;
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
        let metadata = array.metadata_array().filter(mask.clone())?;
        let value = array
            .value_array()
            .map(|v| v.filter(mask.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
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
        let metadata = array.metadata_array().take(indices.clone())?;
        let value = array
            .value_array()
            .map(|v| v.take(indices.clone()))
            .transpose()?;
        let typed_value = array
            .typed_value_array()
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

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StructArray;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ParquetVariantData;

    fn make_unshredded_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(4);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from(99i64));
        ParquetVariantData::from_arrow_variant(&builder.build())
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
        )
        .unwrap();
        let arrow_variant = ArrowVariantArray::try_new(&null_struct).unwrap();
        ParquetVariantData::from_arrow_variant(&arrow_variant)
    }

    #[test]
    fn test_slice_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let sliced = arr.slice(1..3)?;

        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            sliced.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_slice_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        let sliced = arr.slice(0..3)?;

        assert_eq!(sliced.len(), 3);
        assert!(
            !sliced
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            sliced
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !sliced
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_filter_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let mask = Mask::from_iter([true, false, true, false]);
        let filtered = arr.filter(mask)?;

        assert_eq!(filtered.len(), 2);
        assert_eq!(
            filtered.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            filtered.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_filter_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Keep rows 0 (valid), 1 (null), 3 (null)
        let mask = Mask::from_iter([true, true, false, true]);
        let filtered = arr.filter(mask)?;

        assert_eq!(filtered.len(), 3);
        assert!(
            !filtered
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            filtered
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            filtered
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    #[test]
    fn test_take_basic() -> VortexResult<()> {
        let arr = make_unshredded_array()?;
        let indices = PrimitiveArray::from_iter([2u64, 0, 3]);
        let taken = arr.take(indices.into_array())?;

        assert_eq!(taken.len(), 3);
        assert_eq!(
            taken.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            taken.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }

    #[test]
    fn test_take_preserves_validity() -> VortexResult<()> {
        let arr = make_nullable_array()?;
        // Take: valid (0), null (1), null (3), valid (2)
        let indices = PrimitiveArray::from_iter([0u64, 1, 3, 2]);
        let taken = arr.take(indices.into_array())?;

        assert_eq!(taken.len(), 4);
        assert!(
            !taken
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            taken
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            taken
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );
        assert!(
            !taken
                .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_null()
        );

        Ok(())
    }

    fn binary_view_array(values: &[&[u8]]) -> ArrowArrayRef {
        let mut builder = arrow_array::builder::BinaryViewBuilder::new();
        for value in values {
            builder.append_value(*value);
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn test_slice_shredded_typed_value() -> VortexResult<()> {
        let metadata = binary_view_array(&[b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(arrow_array::Int32Array::from(vec![10, 20, 30]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )
        .unwrap();
        let arrow_variant = ArrowVariantArray::try_new(&struct_array).unwrap();
        let arr = ParquetVariantData::from_arrow_variant(&arrow_variant)?;

        let sliced = arr.slice(1..3)?;
        assert_eq!(sliced.len(), 2);
        assert_eq!(
            sliced.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?
        );
        assert_eq!(
            sliced.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            arr.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())?
        );

        Ok(())
    }
}
