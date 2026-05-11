// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::CORE_STORAGE_SLOT;
use crate::arrays::variant::NUM_SLOTS;
use crate::arrays::variant::SHREDDED_SLOT;
use crate::arrays::variant::SLOT_NAMES;
use crate::arrays::variant::VariantArrayExt;
use crate::arrays::variant::compute::rules::RULES;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::matcher::Matcher;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::variant_get::VariantPathElement;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Variant`]-encoded Vortex array.
pub type VariantArray = Array<Variant>;

#[derive(Clone, Debug)]
pub struct Variant;

#[derive(Clone, prost::Message)]
struct VariantMetadataProto {
    #[prost(message, optional, tag = "1")]
    pub shredded_dtype: Option<pb::DType>,
}

impl VTable for Variant {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant");
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
            slots.len() == NUM_SLOTS,
            "VariantArray expects {NUM_SLOTS} slots, got {}",
            slots.len()
        );
        vortex_ensure!(
            slots[CORE_STORAGE_SLOT].is_some(),
            "VariantArray core_storage slot must be present"
        );
        let core_storage = slots[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated core_storage slot presence");
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "Expected Variant DType, got {dtype}"
        );
        vortex_ensure!(
            matches!(core_storage.dtype(), DType::Variant(_)),
            "VariantArray core_storage dtype must be Variant, found {}",
            core_storage.dtype()
        );
        vortex_ensure!(
            core_storage.dtype() == dtype,
            "VariantArray core_storage dtype {} does not match outer dtype {}",
            core_storage.dtype(),
            dtype
        );
        vortex_ensure!(
            core_storage.len() == len,
            "VariantArray core_storage length {} does not match outer length {}",
            core_storage.len(),
            len
        );
        if let Some(shredded) = slots[SHREDDED_SLOT].as_ref() {
            vortex_ensure!(
                shredded.len() == len,
                "VariantArray shredded length {} does not match outer length {}",
                shredded.len(),
                len
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let shredded_dtype = array.slots()[SHREDDED_SLOT]
            .as_ref()
            .map(|shredded| shredded.dtype().try_into())
            .transpose()?;
        Ok(Some(
            VariantMetadataProto { shredded_dtype }.encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "VariantArray expects 0 buffers, got {}",
            buffers.len()
        );
        let proto = VariantMetadataProto::decode(metadata)?;
        let shredded_dtype = proto
            .shredded_dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?;
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let expected_children = 1 + usize::from(shredded_dtype.is_some());
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {} children, got {}",
            expected_children,
            children.len(),
        );
        let core_storage = children.get(0, dtype, len)?;
        let shredded = shredded_dtype
            .map(|dtype| children.get(1, &dtype, len))
            .transpose()?;
        Ok(
            crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(vec![Some(core_storage), shredded].into()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantArray slot_name index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(parent) = ExactScalarFn::<VariantGet>::try_match(parent) else {
            return Ok(None);
        };
        if child_idx != 0 || array.core_storage().is::<Variant>() {
            return Ok(None);
        }

        let typed = array
            .shredded()
            .map(|shredded| {
                typed_shredded_path(shredded, parent.options.path().elements(), ctx)?
                    .map(|typed| typed.mask(array.core_storage().is_not_null()?))
                    .transpose()
            })
            .transpose()?
            .flatten();

        let Some(typed) = typed else {
            return execute_fallback_variant_get(
                array.len(),
                parent.options.clone(),
                array.core_storage().clone(),
                ctx,
            )
            .map(Some);
        };
        if typed.dtype().is_variant()
            && parent
                .options
                .dtype()
                .is_some_and(|dtype| !dtype.is_variant())
        {
            return execute_fallback_variant_get(
                array.len(),
                VariantGetOptions::new(VariantPath::root(), parent.options.dtype().cloned()),
                typed,
                ctx,
            )
            .map(Some);
        }
        if parent.options.dtype().is_none_or(DType::is_variant) {
            let fallback = match typed.dtype() {
                DType::Struct(..) => Some(execute_fallback_variant_get(
                    array.len(),
                    parent.options.clone(),
                    array.core_storage().clone(),
                    ctx,
                )?),
                DType::List(..) | DType::FixedSizeList(..) => {
                    return execute_fallback_variant_get(
                        array.len(),
                        parent.options.clone(),
                        array.core_storage().clone(),
                        ctx,
                    )
                    .map(Some);
                }
                _ if all_valid(&typed, ctx)? => None,
                _ => Some(execute_fallback_variant_get(
                    array.len(),
                    parent.options.clone(),
                    array.core_storage().clone(),
                    ctx,
                )?),
            };
            return merge_typed_as_variant(typed, fallback, ctx).map(Some);
        }

        let requested_dtype = parent
            .options
            .dtype()
            .vortex_expect("variant dtype handled above");
        if typed.dtype().as_nullable() != requested_dtype.as_nullable() {
            return execute_fallback_variant_get(
                array.len(),
                parent.options.clone(),
                array.core_storage().clone(),
                ctx,
            )
            .map(Some);
        }

        let typed = typed.cast(parent.dtype().clone())?;
        if all_valid(&typed, ctx)? {
            return Ok(Some(typed));
        }

        let fallback = execute_fallback_variant_get(
            array.len(),
            parent.options.clone(),
            array.core_storage().clone(),
            ctx,
        )?;
        let typed_mask = typed.is_not_null()?;
        typed_mask
            .zip(typed, fallback)?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

fn typed_shredded_path(
    shredded: &ArrayRef,
    path: &[VariantPathElement],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut current = shredded.clone();
    for element in path {
        let VariantPathElement::Field(name) = element else {
            return Ok(None);
        };
        let DType::Struct(..) = current.dtype() else {
            return Ok(None);
        };
        let current_struct = current.execute::<Array<Struct>>(ctx)?;
        let Some(field) = current_struct.unmasked_field_by_name_opt(name.as_ref()) else {
            return Ok(None);
        };
        current = mask_with_validity(field.clone(), current_struct.validity()?)?;
    }

    Ok(Some(current))
}

fn mask_with_validity(array: ArrayRef, validity: Validity) -> VortexResult<ArrayRef> {
    if validity.no_nulls() {
        return Ok(array);
    }

    let len = array.len();
    array.mask(validity.to_array(len))
}

fn all_valid(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    Ok(array.validity()?.execute_mask(array.len(), ctx)?.all_true())
}

fn merge_typed_as_variant(
    typed: ArrayRef,
    fallback: Option<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Variant(Nullability::Nullable);
    let mut chunks = Vec::with_capacity(typed.len());

    for idx in 0..typed.len() {
        let typed_scalar = typed.execute_scalar(idx, ctx)?;
        let fallback_scalar = fallback
            .as_ref()
            .map(|fallback| fallback.execute_scalar(idx, ctx))
            .transpose()?;
        let scalar = merge_typed_scalar_as_variant(typed_scalar, fallback_scalar, &dtype)?;

        chunks.push(ConstantArray::new(scalar, 1).into_array());
    }

    let core_storage = ChunkedArray::try_new(chunks, dtype)?.into_array();
    VariantArray::try_new(core_storage, None).map(|array| array.into_array())
}

fn merge_typed_scalar_as_variant(
    typed_scalar: Scalar,
    fallback_scalar: Option<Scalar>,
    dtype: &DType,
) -> VortexResult<Scalar> {
    let scalar = if typed_scalar.is_null() {
        fallback_scalar.unwrap_or_else(|| Scalar::null(dtype.clone()))
    } else if matches!(
        typed_scalar.dtype(),
        DType::List(..) | DType::FixedSizeList(..)
    ) {
        Scalar::variant(typed_list_as_variant_payload(typed_scalar)?)
    } else if typed_scalar.dtype().is_struct() {
        merge_typed_object_as_variant(typed_scalar, fallback_scalar)?
    } else if typed_scalar.dtype().is_variant() {
        typed_scalar
    } else {
        Scalar::variant(typed_scalar)
    };

    scalar.cast(dtype)
}

fn typed_list_as_variant_payload(typed_scalar: Scalar) -> VortexResult<Scalar> {
    let list = typed_scalar.as_list();
    let elements = list
        .elements()
        .unwrap_or_default()
        .into_iter()
        .map(|element| {
            if element.dtype().is_variant() {
                element
            } else {
                Scalar::variant(element)
            }
        })
        .collect();
    Ok(Scalar::list(
        DType::Variant(Nullability::NonNullable),
        elements,
        Nullability::NonNullable,
    ))
}

fn merge_typed_object_as_variant(
    typed_scalar: Scalar,
    fallback_scalar: Option<Scalar>,
) -> VortexResult<Scalar> {
    let fallback_inner = fallback_scalar
        .as_ref()
        .and_then(|scalar| scalar.as_variant().value())
        .filter(|scalar| scalar.dtype().is_struct() && !scalar.is_null());
    let Some(fallback_inner) = fallback_inner else {
        return Ok(Scalar::variant(typed_scalar));
    };

    merge_struct_payload(&typed_scalar, Some(fallback_inner)).map(Scalar::variant)
}

fn merge_struct_payload(typed: &Scalar, raw: Option<&Scalar>) -> VortexResult<Scalar> {
    let typed_struct = typed.as_struct();
    let raw_struct = raw
        .filter(|scalar| scalar.dtype().is_struct() && !scalar.is_null())
        .map(Scalar::as_struct);
    let mut present_typed_fields = HashSet::new();
    let mut names = Vec::new();
    let mut values = Vec::new();

    for name in typed_struct.names().iter() {
        let Some(typed_field) = typed_struct.field(name.as_ref()) else {
            continue;
        };
        if typed_field.is_null() {
            continue;
        }

        let raw_field = raw_struct.and_then(|raw_struct| raw_struct.field(name.as_ref()));
        let raw_payload = raw_field.as_ref().and_then(|scalar| {
            if scalar.dtype().is_variant() {
                scalar.as_variant().value()
            } else {
                Some(scalar)
            }
        });
        let field = if typed_field.dtype().is_struct()
            && raw_payload.is_some_and(|raw| raw.dtype().is_struct() && !raw.is_null())
        {
            Scalar::variant(merge_struct_payload(&typed_field, raw_payload)?)
        } else if typed_field.dtype().is_variant() {
            typed_field.cast(&DType::Variant(Nullability::NonNullable))?
        } else {
            Scalar::variant(typed_field)
        };

        present_typed_fields.insert(name.as_ref().to_string());
        names.push(FieldName::from(name.as_ref()));
        values.push(field.into_value());
    }

    if let Some(raw_struct) = raw_struct {
        for name in raw_struct.names().iter() {
            if present_typed_fields.contains(name.as_ref()) {
                continue;
            }
            let Some(raw_field) = raw_struct.field(name.as_ref()) else {
                continue;
            };
            if raw_field.is_null() {
                continue;
            }
            let raw_field = if raw_field.dtype().is_variant() {
                raw_field.cast(&DType::Variant(Nullability::NonNullable))?
            } else {
                Scalar::variant(raw_field)
            };
            names.push(FieldName::from(name.as_ref()));
            values.push(raw_field.into_value());
        }
    }

    let fields = StructFields::new(
        FieldNames::from(names),
        vec![DType::Variant(Nullability::NonNullable); values.len()],
    );
    Scalar::try_new(
        DType::Struct(fields, Nullability::NonNullable),
        Some(ScalarValue::Tuple(values)),
    )
}

fn execute_fallback_variant_get(
    len: usize,
    options: VariantGetOptions,
    core_storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    VariantGet
        .try_new_array(len, options, [core_storage])?
        .execute::<ArrayRef>(ctx)
}

#[cfg(test)]
mod tests {}
