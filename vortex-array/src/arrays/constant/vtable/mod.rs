// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use itertools::Itertools;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::unsupported_buffer_replacement;
use crate::arrays::ExtensionArray;
use crate::arrays::constant::ConstantData;
use crate::arrays::constant::compute::rules::PARENT_RULES;
use crate::arrays::constant::vtable::canonical::constant_canonicalize;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::BoolBuilder;
use crate::builders::DecimalBuilder;
use crate::builders::FixedSizeListBuilder;
use crate::builders::ListViewBuilder;
use crate::builders::NullBuilder;
use crate::builders::PrimitiveBuilder;
use crate::builders::VarBinViewBuilder;
use crate::builders::builder_with_capacity;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::OffsetBuilderPType;
use crate::match_each_decimal_value;
use crate::match_each_listview_builder;
use crate::match_each_native_ptype;
use crate::match_each_varbin_builder;
use crate::scalar::DecimalValue;
use crate::scalar::ListScalar;
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        unsupported_buffer_replacement(array, buffers)
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
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(ArrayParts::new(
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
                if let Some(result) = match_each_varbin_builder!(builder, |builder| {
                    builder.append_scalar_repeated(scalar, n)
                }) {
                    result?;
                } else {
                    append_value_or_nulls::<VarBinViewBuilder>(builder, scalar.is_null(), n, |b| {
                        let value = scalar
                            .as_utf8()
                            .value()
                            .vortex_expect("non-null utf8 scalar must have a value");
                        b.append_n_values(value.as_bytes(), n);
                    });
                }
            }
            DType::Binary(_) => {
                if let Some(result) = match_each_varbin_builder!(builder, |builder| {
                    builder.append_scalar_repeated(scalar, n)
                }) {
                    result?;
                } else {
                    append_value_or_nulls::<VarBinViewBuilder>(builder, scalar.is_null(), n, |b| {
                        let value = scalar
                            .as_binary()
                            .value()
                            .vortex_expect("non-null binary scalar must have a value");
                        b.append_n_values(value, n);
                    });
                }
            }
            DType::List(..) => append_constant_list_run(array, n, builder, ctx)?,
            DType::Extension(ext_dtype) => {
                // An extension array is its storage wearing a dtype, so a run of identical values
                // is a constant storage array, which stays constant-encoded in the builder.
                // Canonicalizing instead would materialize the storage: see the note in
                // `constant_canonicalize` about `ExtensionConstantRule`.
                let storage = ConstantArray::new(scalar.as_extension().to_storage_scalar(), n);
                ExtensionArray::new(ext_dtype.clone(), storage.into_array())
                    .into_array()
                    .append_to_builder(builder, ctx)?
            }
            DType::FixedSizeList(..) => {
                append_constant_fixed_size_list_run(array, n, builder, ctx)?
            }
            // The remaining dtypes canonicalize cheaply: a constant struct canonicalizes to
            // constant fields, and a constant map to views sharing one copy of the entries, so
            // appending the canonical array preserves the run's economy.
            // TODO: add a fast path for DType::Union once it has a builder.
            _ => append_via_canonical(array, builder, ctx)?,
        }

        Ok(())
    }
}

/// Appends the constant list `array` as one run sharing a single copy of its elements.
///
/// The list's elements materialize once, and
/// [`ListViewBuilder::append_array_as_repeated_list`] points the run's `n` views at that one
/// copy. Only a list-view builder has a layout that can share elements; any other builder for a
/// list dtype - a [`ListBuilder`](crate::builders::ListBuilder), whose offsets can only describe
/// contiguous lists - appends the canonical run instead.
fn append_constant_list_run(
    array: ArrayView<'_, Constant>,
    n: usize,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let scalar = array.scalar();
    match match_each_listview_builder!(builder, |b| append_repeated_list_run(
        b,
        scalar.as_list(),
        n,
        ctx
    )) {
        Some(result) => result,
        None => append_via_canonical(array, builder, ctx),
    }
}

/// Appends the list `scalar` to a [`ListViewBuilder`] `n` times, storing its elements once.
fn append_repeated_list_run<O: OffsetBuilderPType, S: OffsetBuilderPType>(
    builder: &mut ListViewBuilder<O, S>,
    scalar: ListScalar,
    n: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if n == 0 {
        return Ok(());
    }

    let Some(elements) = scalar.elements() else {
        // A null run stores no elements at all.
        builder.append_nulls(n);
        return Ok(());
    };

    let mut elements_builder = builder_with_capacity(scalar.element_dtype(), elements.len());
    for element in &elements {
        elements_builder.append_scalar(element)?;
    }

    builder.append_array_as_repeated_list(&elements_builder.finish(), n, ctx)
}

/// Appends the constant fixed-size-list `array` as its list's elements tiled `n` times.
///
/// The list's elements materialize into a tile once - a single [`ConstantArray`] when they are
/// all the same scalar, so that the tiling costs nothing - and
/// [`FixedSizeListBuilder::append_array_as_repeated_list`] shares that one tile across the run.
fn append_constant_fixed_size_list_run(
    array: ArrayView<'_, Constant>,
    n: usize,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let Some(builder) = builder.as_any_mut().downcast_mut::<FixedSizeListBuilder>() else {
        return append_via_canonical(array, builder, ctx);
    };

    if n == 0 {
        return Ok(());
    }

    let scalar = array.scalar().as_list();
    let Some(elements) = scalar.elements() else {
        // A null run stores no elements of its own, only the placeholders the builder writes.
        builder.append_nulls(n);
        return Ok(());
    };

    let tile = match elements.iter().all_equal_value() {
        Ok(uniform) => ConstantArray::new(uniform.clone(), elements.len()).into_array(),
        Err(_) => {
            let mut tile_builder = builder_with_capacity(builder.element_dtype(), elements.len());
            for element in &elements {
                tile_builder.append_scalar(element)?;
            }
            tile_builder.finish()
        }
    };

    builder.append_array_as_repeated_list(&tile, n, ctx)
}

/// Appends `array` by canonicalizing it first, for the dtypes with no fast path of their own.
fn append_via_canonical(
    array: ArrayView<'_, Constant>,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical = array
        .array()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_array();
    canonical.append_to_builder(builder, ctx)
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
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Extension;
    use crate::arrays::FixedSizeList;
    use crate::arrays::ListView;
    use crate::arrays::Struct;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::arrays::constant::vtable::canonical::constant_canonicalize;
    use crate::arrays::extension::ExtensionArrayExt;
    use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::arrays::struct_::StructArrayExt;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
    use crate::builders::ListBuilder;
    use crate::builders::builder_with_capacity;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::Scalar;

    /// Appends `array` into a fresh builder and asserts the result matches `constant_canonicalize`.
    fn assert_append_matches_canonical(array: ConstantArray) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();

        let expected = constant_canonicalize(array.as_view(), &mut ctx)?.into_array();
        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_null_constant_append() -> VortexResult<()> {
        assert_append_matches_canonical(ConstantArray::new(Scalar::null(DType::Null), 5))
    }

    #[test]
    fn test_with_buffers_rejects_serialized_scalar_buffer() {
        let array =
            ConstantArray::new(Scalar::primitive(42i32, Nullability::NonNullable), 3).into_array();
        let buffers = array.buffer_handles();

        // SAFETY: the replacement buffers are the array's existing buffers, so the logical values
        // would be unchanged if the encoding supported buffer replacement.
        let Err(err) = (unsafe { array.with_buffers(buffers) }) else {
            panic!("ConstantArray should reject replacing its serialized scalar buffer");
        };
        assert!(
            err.to_string()
                .contains("does not support in-memory buffer replacement")
        );
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

    #[rstest]
    #[case::non_empty(vec![Scalar::from(1i32), Scalar::from(2i32)], 4)]
    #[case::empty(vec![], 3)]
    #[case::n_zero(vec![Scalar::from(1i32)], 0)]
    fn test_list_constant_append(
        #[case] elements: Vec<Scalar>,
        #[case] n: usize,
    ) -> VortexResult<()> {
        let scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            elements,
            Nullability::NonNullable,
        );
        assert_append_matches_canonical(ConstantArray::new(scalar, n))
    }

    #[test]
    fn test_null_list_constant_append() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert_append_matches_canonical(ConstantArray::new(Scalar::null(dtype), 3))
    }

    /// A run of identical lists appended into a list-view builder shares one copy of its elements
    /// across the whole run.
    #[test]
    fn test_list_constant_append_keeps_one_copy_of_the_elements() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![Scalar::from(1i32), Scalar::from(2i32), Scalar::from(3i32)],
            Nullability::NonNullable,
        );
        let array = ConstantArray::new(scalar, 1_000);

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();

        assert_eq!(
            result.as_::<ListView>().elements().len(),
            3,
            "the run's elements should be stored once, not once per row",
        );
        Ok(())
    }

    /// A `ListBuilder`'s offsets can only describe contiguous lists, so a constant run cannot
    /// share its elements there and takes the canonical path instead.
    #[test]
    fn test_list_constant_append_into_list_builder() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let element_dtype: Arc<DType> =
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let scalar = Scalar::list(
            Arc::clone(&element_dtype),
            vec![Scalar::from(1i32), Scalar::from(2i32)],
            Nullability::NonNullable,
        );
        let array = ConstantArray::new(scalar, 4).into_array();

        let mut builder =
            ListBuilder::<u32>::with_capacity(element_dtype, Nullability::NonNullable, 0, 0);
        array.append_to_builder(&mut builder, &mut ctx)?;

        assert_arrays_eq!(&builder.finish(), &array, &mut ctx);
        Ok(())
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

    /// A run of identical structs should leave each field constant-encoded rather than materialize
    /// a value per row per field.
    #[test]
    fn test_struct_constant_append_keeps_fields_constant() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
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
        let array = ConstantArray::new(scalar, 1_000);

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();

        let struct_array = result.as_::<Struct>();
        for field in 0..2 {
            assert!(
                struct_array.unmasked_field(field).is::<Constant>(),
                "field {field} should have stayed constant-encoded",
            );
        }
        Ok(())
    }

    #[rstest]
    #[case::non_uniform(vec![Scalar::from(1i32), Scalar::from(2i32)])]
    #[case::uniform(vec![Scalar::from(7i32), Scalar::from(7i32)])]
    fn test_fixed_size_list_constant_append(#[case] elements: Vec<Scalar>) -> VortexResult<()> {
        let scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            elements,
            Nullability::NonNullable,
        );
        assert_append_matches_canonical(ConstantArray::new(scalar, 4))
    }

    #[test]
    fn test_null_fixed_size_list_constant_append() -> VortexResult<()> {
        let dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::Nullable,
        );
        assert_append_matches_canonical(ConstantArray::new(Scalar::null(dtype), 3))
    }

    /// A fixed-size list whose elements are all the same scalar tiles a constant array, so the
    /// tile's chunks stay constant-encoded rather than materializing a value per row.
    #[test]
    fn test_uniform_fixed_size_list_constant_append_keeps_elements_constant() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let scalar = Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![Scalar::from(7i32), Scalar::from(7i32)],
            Nullability::NonNullable,
        );
        let array = ConstantArray::new(scalar, 1_000);

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();

        let elements = result.as_::<FixedSizeList>().elements().clone();
        assert!(
            elements
                .as_::<Chunked>()
                .iter_chunks()
                .all(|chunk| chunk.is::<Constant>()),
            "a uniform tile should have stayed constant-encoded",
        );
        Ok(())
    }

    #[test]
    fn test_extension_constant_append() -> VortexResult<()> {
        let scalar = Scalar::extension::<Date>(TimeUnit::Days, Scalar::from(Some(42i32)));
        assert_append_matches_canonical(ConstantArray::new(scalar, 5))
    }

    /// An extension array is its storage wearing a dtype, so a run of identical values should leave
    /// the storage constant-encoded.
    #[test]
    fn test_extension_constant_append_keeps_storage_constant() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let scalar = Scalar::extension::<Date>(TimeUnit::Days, Scalar::from(Some(42i32)));
        let array = ConstantArray::new(scalar, 1_000);

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array
            .into_array()
            .append_to_builder(builder.as_mut(), &mut ctx)?;
        let result = builder.finish();

        assert!(
            result.as_::<Extension>().storage_array().is::<Constant>(),
            "the storage should have stayed constant-encoded",
        );
        Ok(())
    }
}
