// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::constant::ConstantData;
use crate::arrays::constant::compute::rules::PARENT_RULES;
use crate::arrays::constant::vtable::canonical::constant_canonicalize;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::BoolBuilder;
use crate::builders::DecimalBuilder;
use crate::builders::NullBuilder;
use crate::builders::PrimitiveBuilder;
use crate::builders::VarBinViewBuilder;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::match_each_decimal_value;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::serde::ArrayChildren;
pub(crate) mod canonical;
mod operations;
mod validity;

/// A [`Constant`]-encoded Vortex array.
pub type ConstantArray = Array<Constant>;

#[derive(Clone, Debug)]
pub struct Constant;

impl ArrayHash for ConstantData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.scalar.hash(state);
    }
}

impl ArrayEq for ConstantData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.scalar == other.scalar
    }
}

impl VTable for Constant {
    type TypedArrayData = ConstantData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.constant");
        *ID
    }

    fn validate(
        &self,
        data: &ConstantData,
        dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            data.scalar.dtype() == dtype,
            "ConstantArray scalar dtype does not match outer dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(
                ScalarValue::to_proto_bytes::<ByteBufferMut>(array.scalar.value()).freeze(),
            ),
            _ => vortex_panic!("ConstantArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("scalar".to_string()),
            _ => None,
        }
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("ConstantArray slot_name index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // HACK: Because the scalar is stored in the buffers, we do not need to serialize the
        // metadata at all.
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &[u8],

        buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(crate::array::ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            ConstantData::new(scalar),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(constant_canonicalize(
            array.as_view(),
            ctx,
        )?))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let n = array.len();
        let scalar = array.scalar();

        match array.dtype() {
            DType::Null => append_value_or_nulls::<NullBuilder>(builder, true, n, |_| {}),
            DType::Bool(_) => {
                append_value_or_nulls::<BoolBuilder>(builder, scalar.is_null(), n, |b| {
                    b.append_values(
                        scalar
                            .as_bool()
                            .value()
                            .vortex_expect("non-null bool scalar must have a value"),
                        n,
                    );
                })
            }
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |P| {
                    append_value_or_nulls::<PrimitiveBuilder<P>>(
                        builder,
                        scalar.is_null(),
                        n,
                        |b| {
                            let value = P::try_from(scalar)
                                .vortex_expect("Couldn't unwrap constant scalar to primitive");
                            b.append_n_values(value, n);
                        },
                    );
                });
            }
            DType::Decimal(..) => {
                append_value_or_nulls::<DecimalBuilder>(builder, scalar.is_null(), n, |b| {
                    let value = scalar
                        .as_decimal()
                        .decimal_value()
                        .vortex_expect("non-null decimal scalar must have a value");
                    match_each_decimal_value!(value, |v| { b.append_n_values(v, n) });
                });
            }
            DType::Utf8(_) => {
                append_value_or_nulls::<VarBinViewBuilder>(builder, scalar.is_null(), n, |b| {
                    let typed = scalar.as_utf8();
                    let value = typed
                        .value()
                        .vortex_expect("non-null utf8 scalar must have a value");
                    b.append_n_values(value.as_bytes(), n);
                });
            }
            DType::Binary(_) => {
                append_value_or_nulls::<VarBinViewBuilder>(builder, scalar.is_null(), n, |b| {
                    let typed = scalar.as_binary();
                    let value = typed
                        .value()
                        .vortex_expect("non-null binary scalar must have a value");
                    b.append_n_values(value, n);
                });
            }
            // TODO: add fast paths for DType::Struct, DType::List, DType::FixedSizeList, DType::Extension.
            _ => {
                let canonical = array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array();
                builder.extend_from_array(&canonical);
            }
        }

        Ok(())
    }
}

/// Downcasts `builder` to `B`, then either appends `n` nulls or calls `fill` with the typed
/// builder depending on `is_null`.
///
/// `is_null` must only be `true` when the builder is nullable.
fn append_value_or_nulls<B: ArrayBuilder + 'static>(
    builder: &mut dyn ArrayBuilder,
    is_null: bool,
    n: usize,
    fill: impl FnOnce(&mut B),
) {
    let b = builder
        .as_any_mut()
        .downcast_mut::<B>()
        .vortex_expect("builder dtype must match array dtype");
    if is_null {
        // SAFETY: is_null=true only when the scalar (and thus the builder) is nullable.
        unsafe { b.append_nulls_unchecked(n) };
    } else {
        fill(b);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::constant::vtable::canonical::constant_canonicalize;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::scalar::Scalar;

    /// Appends `array` into a fresh builder and asserts the result matches `constant_canonicalize`.
    fn assert_append_matches_canonical(array: ConstantArray) -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();

        let expected = constant_canonicalize(array.as_view(), &mut ctx)?.into_array();
        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn test_null_constant_append() -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(Scalar::null(DType::Null), 5))
    }

    #[rstest]
    #[case::bool_true(true, 5)]
    #[case::bool_false(false, 3)]
    fn test_bool_constant_append(#[case] value: bool, #[case] n: usize) -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::bool(value, Nullability::NonNullable),
            n,
        ))
    }

    #[test]
    fn test_bool_null_constant_append() -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            4,
        ))
    }

    #[rstest]
    #[case::i32(Scalar::primitive(42i32, Nullability::NonNullable), 5)]
    #[case::u8(Scalar::primitive(7u8, Nullability::NonNullable), 3)]
    #[case::f64(Scalar::primitive(1.5f64, Nullability::NonNullable), 4)]
    #[case::i32_null(Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)), 3)]
    fn test_primitive_constant_append(
        #[case] scalar: Scalar,
        #[case] n: usize,
    ) -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(scalar, n))
    }

    #[rstest]
    #[case::utf8_inline("hi", 5)] // ≤12 bytes: inlined in BinaryView
    #[case::utf8_noninline("hello world!!", 5)] // >12 bytes: requires buffer block
    #[case::utf8_empty("", 3)]
    #[case::utf8_n_zero("hello world!!", 0)] // n=0 with non-inline: must not write orphaned bytes
    fn test_utf8_constant_append(#[case] value: &str, #[case] n: usize) -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::utf8(value, Nullability::NonNullable),
            n,
        ))
    }

    #[test]
    fn test_utf8_null_constant_append() -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::null(DType::Utf8(Nullability::Nullable)),
            4,
        ))
    }

    #[rstest]
    #[case::binary_inline(vec![1u8, 2, 3], 5)] // ≤12 bytes: inlined
    #[case::binary_noninline(vec![0u8; 13], 5)] // >12 bytes: buffer block
    fn test_binary_constant_append(#[case] value: Vec<u8>, #[case] n: usize) -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::binary(value, Nullability::NonNullable),
            n,
        ))
    }

    #[test]
    fn test_binary_null_constant_append() -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(
            Scalar::null(DType::Binary(Nullability::Nullable)),
            4,
        ))
    }

    #[test]
    fn test_struct_constant_append() -> VortexResult<()> {
        let fields = StructFields::new(
            ["x", "y"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        );
        let scalar = Scalar::struct_(
            DType::Struct(fields, Nullability::NonNullable),
            [
                Scalar::primitive(42i32, Nullability::NonNullable),
                Scalar::utf8("hi", Nullability::NonNullable),
            ],
        );
        assert_append_matches_canonical(ConstantArray::new(scalar, 3))
    }

    #[test]
    fn test_null_struct_constant_append() -> VortexResult<()> {
        let fields = StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
        );
        let dtype = DType::Struct(fields, Nullability::Nullable);
        assert_append_matches_canonical(ConstantArray::new(Scalar::null(dtype), 4))
    }
}
