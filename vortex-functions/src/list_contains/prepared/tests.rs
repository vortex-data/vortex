// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::runend::RunEndArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::builtin::operators::CompareMode;
use vortex_array::scalar_fn::builtin::operators::FloatOrder;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_expr::ArrayExprExt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::PreparedSet;
use crate::expr::list_contains;
use crate::list_contains::ListContains;
use crate::list_contains::ListContainsOptions;
use crate::list_contains::PreparedSetData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

const SQL: ListContainsOptions = ListContainsOptions {
    sql_null_semantics: true,
    compare_mode: CompareMode::new(FloatOrder::ZerosEqual),
};

fn set_with_null() -> Scalar {
    let element = DType::Primitive(PType::I32, Nullability::Nullable);
    Scalar::list(
        element.clone(),
        vec![
            Scalar::primitive(2i32, Nullability::Nullable),
            Scalar::null(element),
        ],
        Nullability::NonNullable,
    )
}

fn empty_i32_set() -> Scalar {
    Scalar::list(
        DType::Primitive(PType::I32, Nullability::Nullable),
        vec![],
        Nullability::NonNullable,
    )
}

fn list_set(node: &ArrayRef) -> PreparedSetData {
    PreparedSet::set_of(node.as_::<ScalarFn>().get_child(0))
        .vortex_expect("the list of the node is a prepared set")
        .clone()
}

fn chunked_needles() -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array(),
            PrimitiveArray::from_option_iter::<i32, _>([]).into_array(),
            PrimitiveArray::from_option_iter([None, Some(3), Some(2)]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::Nullable),
    )?
    .into_array())
}

#[test]
fn test_prepared_set_shares_haystack_across_slice_filter_and_zero_child_guard() -> VortexResult<()>
{
    let mut ctx = SESSION.create_execution_ctx();
    let list = set_with_null();
    let scalar_fn =
        TypedScalarFnInstance::new(PreparedSet, PreparedSetData::try_new(list.clone())?).erased();

    let err = ScalarFnArray::try_new(scalar_fn.clone(), vec![]).unwrap_err();
    assert!(
        err.to_string()
            .contains("ScalarFnArray length cannot be inferred without children"),
        "{err}"
    );

    let set = ScalarFnArray::try_new_with_len(scalar_fn, vec![], 4)?.into_array();
    assert_eq!(set.as_::<ScalarFn>().nchildren(), 0);
    assert_arrays_eq!(set, ConstantArray::new(list.clone(), 4), &mut ctx);

    let prepared = PreparedSet::set_of(&set).vortex_expect("prepared set");
    let sliced = set.slice(1..3)?.optimize_ctx(&SESSION)?;
    assert_eq!(sliced.len(), 2);
    assert!(PreparedSet::set_of(&sliced).is_some_and(|s| s.shares_set_with(prepared)));

    let filtered = FilterArray::try_new(set.clone(), Mask::from_iter([true, false, true, true]))?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert_eq!(filtered.len(), 3);
    assert_eq!(filtered.as_::<ScalarFn>().nchildren(), 0);
    assert!(PreparedSet::set_of(&filtered).is_some_and(|s| s.shares_set_with(prepared)));
    assert_arrays_eq!(filtered, ConstantArray::new(list, 3), &mut ctx);
    Ok(())
}

#[rstest]
#[case::array_default(false, ListContainsOptions::default(), [Some(false), Some(true), None, Some(false), Some(true)])]
#[case::array_sql(false, ListContainsOptions { sql_null_semantics: true, ..Default::default() }, [None, Some(true), None, None, Some(true)])]
#[case::expr_default(true, ListContainsOptions::default(), [Some(false), Some(true), None, Some(false), Some(true)])]
fn test_chunked_pushdown_shares_one_prepared_set(
    #[case] via_expr: bool,
    #[case] options: ListContainsOptions,
    #[case] expected: [Option<bool>; 5],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let needles = chunked_needles()?;
    let array = if via_expr {
        let expr = list_contains(lit(set_with_null()), root())
            .bind(needles.dtype())?
            .optimize()?;
        needles.apply_bound(&expr)?.optimize_ctx(&SESSION)?
    } else {
        let list = ConstantArray::new(set_with_null(), needles.len()).into_array();
        ListContains::try_new_opts(list, needles, options)?
            .into_array()
            .optimize_ctx(&SESSION)?
    };

    let sets: Vec<_> = array
        .as_::<Chunked>()
        .iter_chunks()
        .filter(|chunk| !chunk.is_empty())
        .map(list_set)
        .collect();
    assert!(sets.len() > 1);
    assert!(sets.iter().all(|s| s.shares_set_with(&sets[0])));
    assert_arrays_eq!(array, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[rstest]
#[case::null_code_default(true, ListContainsOptions::default(), set_with_null(), vec![Some(true), Some(false), None, Some(true)])]
#[case::null_code_sql(true, SQL, set_with_null(), vec![Some(true), None, None, Some(true)])]
#[case::empty_default(true, ListContainsOptions::default(), empty_i32_set(), vec![Some(false); 4])]
#[case::empty_sql(true, SQL, empty_i32_set(), vec![Some(false), Some(false), None, Some(false)])]
#[case::null_value_default(false, ListContainsOptions::default(), set_with_null(), vec![Some(true), Some(false), Some(true), None, Some(false)])]
fn test_dict_pushdown_structure(
    #[case] null_code: bool,
    #[case] options: ListContainsOptions,
    #[case] set: Scalar,
    #[case] expected: Vec<Option<bool>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let needles = if null_code {
        DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u8), Some(1), None, Some(0)]).into_array(),
            buffer![2i32, 3].into_array(),
        )?
        .into_array()
    } else {
        DictArray::try_new(
            buffer![0u8, 1, 0, 2, 1].into_array(),
            PrimitiveArray::from_option_iter([Some(2i32), Some(3), None]).into_array(),
        )?
        .into_array()
    };
    let list = ConstantArray::new(set, needles.len()).into_array();
    let array = ListContains::try_new_opts(list, needles, options)?
        .into_array()
        .optimize_ctx(&SESSION)?;

    // Rewritten to Dict of answers or Constant for empty list off SQL null semantics.
    assert!(!array.is::<ScalarFn>());
    if let Some(dict) = array.as_opt::<Dict>() {
        assert!(PreparedSet::set_of(dict.values().as_::<ScalarFn>().get_child(0)).is_some());
    }
    assert_arrays_eq!(array, BoolArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn test_runend_pushdown_structure() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let needles = RunEnd::try_new(
        buffer![2u32, 5, 6].into_array(),
        buffer![1i32, 2, 3].into_array(),
        &mut ctx,
    )?
    .into_array();
    let list = ConstantArray::new(
        Scalar::list(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            vec![Scalar::from(2i32), Scalar::from(3i32)],
            Nullability::NonNullable,
        ),
        needles.len(),
    )
    .into_array();

    let array = ListContains::try_new(list, needles)?
        .into_array()
        .optimize_ctx(&SESSION)?;
    assert!(array.is::<RunEnd>());
    let values = array.as_::<RunEnd>().values();
    assert!(PreparedSet::set_of(values.as_::<ScalarFn>().get_child(0)).is_some());
    assert_arrays_eq!(
        array,
        BoolArray::from_iter([false, false, true, true, true, true]),
        &mut ctx
    );
    Ok(())
}

/// Needles longer than one lookup batch (`NEEDLE_BATCH_LEN = 8192`) in a non-primitive encoding
/// decode and probe one batch at a time.
#[test]
fn test_compressed_needles_in_batches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let set = PreparedSetData::try_new(Scalar::list(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        vec![
            Scalar::from(3i64),
            Scalar::from(9000i64),
            Scalar::from(20_001i64),
        ],
        Nullability::NonNullable,
    ))?;
    let values = |range: std::ops::Range<i64>| {
        PrimitiveArray::from_option_iter(range.map(|v| (v % 7 != 0).then_some(v))).into_array()
    };
    let needles = ChunkedArray::try_new(
        vec![values(0..12_000), values(12_000..25_000)],
        DType::Primitive(PType::I64, Nullability::Nullable),
    )?
    .into_array();
    let expected = BoolArray::from_iter(
        (0..25_000i64).map(|v| (v % 7 != 0).then_some(matches!(v, 3 | 9000 | 20_001))),
    );
    assert_arrays_eq!(
        set.contains(&needles, &ListContainsOptions::default(), &mut ctx)?,
        expected,
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_constant_needle_executes_to_constant_for_primitive_and_row_haystacks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Primitive haystack: `3 IN (2, NULL)` -> false (default) or null (SQL).
    let set_arr = PreparedSet::try_new_array(set_with_null(), 4)?;
    let needle = ConstantArray::new(Scalar::primitive(3i32, Nullability::Nullable), 4).into_array();
    for (options, expected) in [
        (
            ListContainsOptions::default(),
            Scalar::bool(false, Nullability::Nullable),
        ),
        (
            ListContainsOptions {
                sql_null_semantics: true,
                ..Default::default()
            },
            Scalar::null(DType::Bool(Nullability::Nullable)),
        ),
    ] {
        let res = ListContains::try_new_opts(set_arr.clone(), needle.clone(), options)?
            .into_array()
            .execute::<ArrayRef>(&mut ctx)?;
        assert_eq!(res.as_constant(), Some(&expected));
    }

    // Row haystack: constant nested list needle looks up one row and yields a constant.
    let nested = ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(1i32), None]).into_array(),
        buffer![0u32, 2].into_array(),
        Validity::AllValid,
        &mut ctx,
    )?
    .into_array();
    let member = nested.execute_scalar(0, &mut ctx)?;
    let list = Scalar::list(
        member.dtype().clone(),
        vec![member.clone()],
        Nullability::NonNullable,
    );
    let res = ListContains::try_new_opts(
        PreparedSet::try_new_array(list, 4)?,
        ConstantArray::new(member, 4).into_array(),
        ListContainsOptions::default(),
    )?
    .into_array()
    .execute::<ArrayRef>(&mut ctx)?;
    assert_eq!(
        res.as_constant(),
        Some(&Scalar::bool(true, Nullability::Nullable))
    );
    Ok(())
}

#[rstest]
#[case::default(ListContainsOptions::default(), [Some(true), Some(false), None])]
#[case::sql(
    ListContainsOptions { sql_null_semantics: true, ..Default::default() },
    [Some(true), None, None],
)]
fn test_result_from_bits_and_guards(
    #[case] options: ListContainsOptions,
    #[case] expected: [Option<bool>; 3],
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let set = PreparedSetData::try_new(set_with_null())?;
    let needle_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let bits = BitBuffer::from_iter([true, false, true]);
    let validity = Validity::from_iter([true, true, false]);

    assert_arrays_eq!(
        set.result_from_bits(bits.clone(), validity, &needle_dtype, &options)?,
        BoolArray::from_iter(expected),
        &mut ctx
    );

    // Mismatched validity length and mismatched needle dtype must fail.
    let short_validity = Validity::from_iter([true, false]);
    assert!(
        set.result_from_bits(bits.clone(), short_validity, &needle_dtype, &options)
            .is_err()
    );
    let wrong_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    assert!(
        set.result_from_bits(bits, Validity::AllValid, &wrong_dtype, &options)
            .is_err()
    );
    Ok(())
}

/// An extension type whose metadata serializer always fails, standing in for a plugin.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct FailingMetadata;

impl ExtVTable for FailingMetadata {
    type Metadata = usize;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a str;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.failing_metadata")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        vortex_bail!("metadata serializer failed")
    }

    fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::StoredMetadata> {
        Ok(0)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        _storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok("")
    }
}

#[test]
fn serialize_propagates_extension_metadata_error() -> VortexResult<()> {
    let scalar =
        Scalar::extension::<FailingMetadata>(0, Scalar::primitive(1i32, Nullability::NonNullable));
    let list = Scalar::list(
        scalar.dtype().clone(),
        vec![scalar],
        Nullability::NonNullable,
    );
    let set = PreparedSetData::try_new(list)?;
    let err = PreparedSet
        .serialize(&set)
        .expect_err("serializing an extension whose metadata serializer fails must fail");
    assert!(
        err.to_string().contains("metadata serializer failed"),
        "{err}"
    );
    Ok(())
}
