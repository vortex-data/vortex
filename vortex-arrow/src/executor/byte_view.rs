// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteViewArray;
use arrow_array::types::ByteViewType;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrowExportOptions;
use crate::CompactBuffers;
use crate::dtype::from_arrow_data_type;
use crate::null_buffer::to_null_buffer;

/// Convert a canonical VarBinViewArray directly to Arrow.
pub fn canonical_varbinview_to_arrow<T: ByteViewType>(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let views = Buffer::<u128>::from_byte_buffer(array.views_handle().as_host().clone())
        .into_arrow_scalar_buffer();
    let buffers: Vec<_> = array
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().clone().into_arrow_buffer())
        .collect();
    let nulls = to_null_buffer(
        array
            .as_ref()
            .validity()?
            .execute_mask(array.as_ref().len(), ctx)?,
    );

    // SAFETY: our own VarBinView array is considered safe.
    Ok(Arc::new(unsafe {
        GenericByteViewArray::<T>::new_unchecked(views, buffers, nulls)
    }))
}

pub fn execute_varbinview_to_arrow<T: ByteViewType>(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    execute_varbinview_to_arrow_with_options::<T>(array, &ArrowExportOptions::default(), ctx)
}

/// Convert a canonical string or binary view array with explicit export options.
pub fn execute_varbinview_to_arrow_with_options<T: ByteViewType>(
    array: &VarBinViewArray,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    if options
        .get::<CompactBuffers>()
        .copied()
        .unwrap_or_default()
        .0
    {
        let compacted = array.compact_buffers(ctx)?;
        canonical_varbinview_to_arrow::<T>(&compacted, ctx)
    } else {
        canonical_varbinview_to_arrow::<T>(array, ctx)
    }
}

pub(super) fn to_arrow_byte_view<T: ByteViewType>(
    array: ArrayRef,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the array into the desired ByteView type.
    // We do this in case the vortex array is Utf8, and we want Binary or vice versa. By casting
    // first, we may push this down through the Vortex array tree. We choose nullable to be most
    // flexible since there's no prescribed nullability in Arrow types.
    let array = array.cast(from_arrow_data_type(&T::DATA_TYPE, Nullability::Nullable)?)?;

    let array = array.execute::<ArrayRef>(ctx)?;
    let varbinview = array.execute::<VarBinViewArray>(ctx)?;
    execute_varbinview_to_arrow_with_options::<T>(&varbinview, options, ctx)
}

#[cfg(test)]
mod tests {
    use arrow_array::RunArray;
    use arrow_array::cast::AsArray;
    use arrow_array::types::BinaryViewType;
    use arrow_array::types::Int32Type;
    use arrow_array::types::StringViewType;
    use arrow_array::types::UInt8Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_runend::RunEnd;

    use super::*;
    use crate::ArrowSessionExt;

    #[test]
    fn empty_views_are_aligned() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(std::iter::empty::<&str>());
        let mut ctx = array_session().create_execution_ctx();

        let arrow = canonical_varbinview_to_arrow::<StringViewType>(&array, &mut ctx)?;

        assert!(arrow.is_empty());
        Ok(())
    }

    #[rstest]
    #[case::flat("flat")]
    #[case::list("list")]
    #[case::list_view("list_view")]
    #[case::fixed_size_list("fixed_size_list")]
    #[case::run_end("run_end")]
    #[case::struct_("struct")]
    #[case::nested("nested")]
    #[case::dictionary("dictionary")]
    fn export_compaction_policy(
        #[case] shape: &str,
        #[values(false, true)] binary: bool,
        #[values(None, Some(false), Some(true))] compact: Option<bool>,
    ) -> VortexResult<()> {
        let session = array_session();
        let options = compact.map_or_else(ArrowExportOptions::default, |compact| {
            ArrowExportOptions::default().with(CompactBuffers(compact))
        });
        let mut ctx = session.create_execution_ctx();
        let value = "x".repeat(256);
        let values = (0..128).map(|i| (i != 1).then(|| format!("{i}{value}")));
        // Keep distant views in one buffer so compaction must copy rather than just trim it.
        let array = if binary {
            VarBinViewArray::from_iter_nullable_bin(values)
        } else {
            VarBinViewArray::from_iter_nullable_str(values)
        }
        .into_array()
        .take(buffer![0u32, 1, 12].into_array())?;
        let view = array.clone().execute::<VarBinViewArray>(&mut ctx)?;
        let backing = view.data_buffers()[0].as_host();
        let expected = if binary {
            canonical_varbinview_to_arrow::<BinaryViewType>(&view, &mut ctx)?
        } else {
            canonical_varbinview_to_arrow::<StringViewType>(&view, &mut ctx)?
        };
        let leaf_type = expected.data_type().clone();
        let field = Arc::new(Field::new("item", leaf_type.clone(), true));
        let (array, target) = match shape {
            "flat" => (array, leaf_type),
            "list" => (
                ListArray::try_new(array, buffer![0i32, 3].into_array(), Validity::NonNullable)?
                    .into_array(),
                DataType::List(field),
            ),
            "fixed_size_list" => (
                FixedSizeListArray::try_new(array, 3, Validity::NonNullable, 1)?.into_array(),
                DataType::FixedSizeList(field, 3),
            ),
            "run_end" => (
                RunEnd::try_new(buffer![1u32, 2, 3].into_array(), array, &mut ctx)?.into_array(),
                DataType::RunEndEncoded(
                    Arc::new(Field::new("ends", DataType::Int32, false)),
                    field,
                ),
            ),
            "list_view" => (
                ListViewArray::try_new(
                    array,
                    buffer![0i32].into_array(),
                    buffer![3i32].into_array(),
                    Validity::NonNullable,
                )?
                .into_array(),
                DataType::ListView(field),
            ),
            "struct" | "nested" => {
                let array = StructArray::try_new(
                    FieldNames::from(["item"]),
                    vec![array],
                    3,
                    Validity::NonNullable,
                )?
                .into_array();
                let target = DataType::Struct(vec![field].into());
                if shape == "nested" {
                    (
                        ListArray::try_new(
                            array,
                            buffer![0i32, 3].into_array(),
                            Validity::NonNullable,
                        )?
                        .into_array(),
                        DataType::List(Arc::new(Field::new("item", target, false))),
                    )
                } else {
                    (array, target)
                }
            }
            "dictionary" => (
                DictArray::try_new(buffer![0u8, 1, 2].into_array(), array)?.into_array(),
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(leaf_type)),
            ),
            _ => unreachable!(),
        };
        let arrow = session.arrow().execute_arrow_with_options(
            array.clone(),
            Some(&Field::new("", target, true)),
            &options,
            &mut ctx,
        )?;
        let leaf = match shape {
            "flat" => arrow,
            "list" => Arc::clone(arrow.as_list::<i32>().values()),
            "list_view" => Arc::clone(arrow.as_list_view::<i32>().values()),
            "fixed_size_list" => Arc::clone(arrow.as_fixed_size_list().values()),
            "run_end" => Arc::clone(
                arrow
                    .as_any()
                    .downcast_ref::<RunArray<Int32Type>>()
                    .expect("run array")
                    .values(),
            ),
            "struct" => Arc::clone(arrow.as_struct().column(0)),
            "nested" => Arc::clone(arrow.as_list::<i32>().values().as_struct().column(0)),
            "dictionary" => Arc::clone(arrow.as_dictionary::<UInt8Type>().values()),
            _ => unreachable!(),
        };
        assert_eq!(leaf.to_data(), expected.to_data());
        let buffers = if binary {
            leaf.as_binary_view().data_buffers()
        } else {
            leaf.as_string_view().data_buffers()
        };
        let retains_backing = buffers
            .iter()
            .any(|buffer| buffer.as_ptr() == backing.as_ptr());
        assert_eq!(retains_backing, compact == Some(false));
        if shape == "flat" && compact == Some(false) {
            let default = session.arrow().execute_arrow(array, None, &mut ctx)?;
            let buffers = if binary {
                default.as_binary_view().data_buffers()
            } else {
                default.as_string_view().data_buffers()
            };
            assert!(
                buffers
                    .iter()
                    .all(|buffer| buffer.as_ptr() != backing.as_ptr())
            );
        }
        Ok(())
    }
}
