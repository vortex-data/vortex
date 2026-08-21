// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Snapshot tests tracing how TPC-H style scan operations reduce and execute over
//! BtrBlocks-compressed lineitem columns.
//!
//! The lineitem table (SF 0.001, deterministic) compresses to the same encodings a real
//! file scan would see: bitpacked integers, `decimal_byte_parts` decimals with dictionary
//! or bitpacked byte parts, `ext(date)` over FoR + bitpacking, dictionary-of-FSST for
//! low-cardinality strings, and FSST for comments. The tests below pin down which reduce
//! rules and execute kernels fire for the scan operations TPC-H queries perform over those
//! encodings.

use std::sync::LazyLock;

use arrow_array::RecordBatch;
use tpchgen::distribution::Distributions;
use tpchgen::generators::LineItemGenerator;
use tpchgen::text::TextPool;
use tpchgen_arrow::LineItemArrow;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_array::test_harness::trace::Traced;
use vortex_array::test_harness::trace::trace_op;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

/// A session with the default Vortex encodings registered.
///
/// Reduce-parent and execute-parent kernels live in the session registry, so an encoding whose
/// `initialize` was never called still decodes correctly but silently contributes none of its
/// pushdown kernels — the traces below would degrade to plain canonicalization without failing
/// any value assertion.
///
/// This mirrors `vortex_file::register_default_encodings`, copied rather than called so that
/// `vortex-btrblocks` does not depend on `vortex-file` (which depends on this crate). Keep the
/// two in step when encodings are added. `bytebool` and `tensor` are the only entries omitted:
/// this crate does not depend on them, so the compressor cannot emit them.
fn trace_session() -> VortexSession {
    let session = VortexSession::empty().with::<ArraySession>();

    vortex_fsst::initialize(&session);
    #[cfg(feature = "unstable_encodings")]
    vortex_onpair::initialize(&session);
    vortex_zigzag::initialize(&session);

    {
        let arrays = session.arrays();
        #[cfg(feature = "pco")]
        arrays.register(vortex_pco::Pco);
        #[cfg(feature = "zstd")]
        arrays.register(vortex_zstd::Zstd);
        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        arrays.register(vortex_zstd::ZstdBuffers);
        if use_experimental_patches() {
            arrays.register(Patched);
        }
    }

    vortex_alp::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    #[cfg(feature = "unstable_encodings")]
    vortex_elias_fano::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sequence::initialize(&session);
    vortex_sparse::initialize(&session);

    session
}

fn execution_ctx() -> ExecutionCtx {
    ExecutionCtx::new(trace_session())
}

/// A 1MiB TPC-H text pool. The spec-default 300MiB pool takes several seconds to initialize in
/// debug builds; a smaller pool keeps the same text distribution and is deterministic.
static TEXT_POOL: LazyLock<TextPool> =
    LazyLock::new(|| TextPool::new(1 << 20, Distributions::static_default()));

/// The first 4096 rows of TPC-H lineitem at scale factor 0.001. Both tpchgen and the
/// compressor (fixed sampling seed) are deterministic, so the resulting encodings are stable.
fn lineitem() -> VortexResult<ArrayRef> {
    let generator = LineItemGenerator::new_with_distributions_and_text_pool(
        0.001,
        1,
        1,
        Distributions::static_default(),
        &TEXT_POOL,
    );
    let batch: RecordBatch = LineItemArrow::new(generator)
        .with_batch_size(1 << 12)
        .next()
        .expect("at least one batch");
    let schema = batch.schema();
    trace_session()
        .arrow()
        .from_arrow_record_batch(batch, &schema)
}

fn compressed_lineitem() -> VortexResult<ArrayRef> {
    BtrBlocksCompressor::default().compress(&lineitem()?, &mut execution_ctx())
}

fn field(array: &ArrayRef, name: &str) -> VortexResult<ArrayRef> {
    Ok(array.as_::<Struct>().unmasked_field_by_name(name)?.clone())
}

/// Trace the optimize pass and then the execution of `array`, asserting that the canonical
/// result matches running the same operation over the uncompressed column.
fn optimize_then_execute(
    array: ArrayRef,
    expected: &ArrayRef,
) -> VortexResult<[Traced<ArrayRef>; 2]> {
    let optimized = trace_op(|| array.optimize())?;
    let optimized_output = optimized.output.clone();
    let executed = trace_op(|| {
        optimized_output
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;
    let expected = expected
        .clone()
        .execute::<Canonical>(&mut execution_ctx())?
        .into_array();
    assert_arrays_eq!(executed.output, expected, &mut execution_ctx());
    Ok([
        Traced {
            output: optimized.output,
            trace: optimized.trace,
        },
        executed,
    ])
}

/// Q6-style predicate over the shipdate column: `l_shipdate >= DATE '1994-01-01'`.
///
/// The column compresses to `ext(date) -> for -> bitpacked`.
fn shipdate_predicate(column: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
    let DType::Extension(ext) = column.dtype().clone() else {
        panic!("expected extension dtype for l_shipdate")
    };
    // 8766 days since the epoch = 1994-01-01.
    let cutoff = Scalar::extension_ref(
        ext,
        Scalar::primitive_value(PValue::I32(8766), PType::I32, Nullability::NonNullable),
    );
    Binary.try_new_array(
        len,
        Operator::Gte,
        [column, ConstantArray::new(cutoff, len).into_array()],
    )
}

#[test]
fn trace_scan_compare_on_compressed_shipdate() -> VortexResult<()> {
    let compressed = compressed_lineitem()?;
    let len = compressed.len();

    let lazy = shipdate_predicate(field(&compressed, "l_shipdate")?, len)?;
    let expected = shipdate_predicate(field(&lineitem()?, "l_shipdate")?, len)?;
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    // No reduce rule rewrites a comparison over the extension array; the extension compare
    // kernel handles it at execution time by comparing the underlying storage.
    insta::assert_snapshot!(optimized.trace.to_string(), @"");
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.binary(bool, len=4096)
      iter 0 current=vortex.binary(bool, len=4096) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.binary(bool, len=4096) child=vortex.ext(vortex.date[days](i32), len=4096) -> vortex.binary(bool, len=4096)
      iter 1 current=vortex.binary(bool, len=4096) builder_active=false
    execute_until target=AnyCanonical root=fastlanes.for(i32, len=4096)
      iter 0 current=fastlanes.for(i32, len=4096) builder_active=false
    execute_until target=AnyCanonical root=fastlanes.bitpacked(i32, len=4096)
      iter 0 current=fastlanes.bitpacked(i32, len=4096) builder_active=false
        Done array=vortex.primitive(i32, len=4096)
      iter 1 current=vortex.primitive(i32, len=4096) builder_active=false
      return output=vortex.primitive(i32, len=4096)
        Done array=vortex.primitive(i32, len=4096)
      iter 1 current=vortex.primitive(i32, len=4096) builder_active=false
      return output=vortex.primitive(i32, len=4096)
        Done array=vortex.bool(bool, len=4096)
      iter 2 current=vortex.bool(bool, len=4096) builder_active=false
      return output=vortex.bool(bool, len=4096)
    ");

    Ok(())
}

/// Q6-style predicate over the quantity column: `l_quantity < 24`.
///
/// The column compresses to `decimal_byte_parts -> dict -> bitpacked/sequence`.
fn quantity_predicate(column: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
    let cutoff = Scalar::decimal(
        DecimalValue::I128(2400),
        DecimalDType::new(15, 2),
        Nullability::NonNullable,
    );
    Binary.try_new_array(
        len,
        Operator::Lt,
        [column, ConstantArray::new(cutoff, len).into_array()],
    )
}

#[test]
fn trace_scan_compare_on_compressed_quantity() -> VortexResult<()> {
    let compressed = compressed_lineitem()?;
    let len = compressed.len();

    let lazy = quantity_predicate(field(&compressed, "l_quantity")?, len)?;
    let expected = quantity_predicate(field(&lineitem()?, "l_quantity")?, len)?;
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    // No reduce rule rewrites a comparison over decimal_byte_parts; its compare kernel fires
    // at execution time and pushes the comparison into the byte-parts dictionary, whose values
    // are then compared and decoded.
    insta::assert_snapshot!(optimized.trace.to_string(), @"");
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.binary(bool, len=4096)
      iter 0 current=vortex.binary(bool, len=4096) builder_active=false
    optimize root=vortex.binary(bool, len=4096) session=false
      reduce_parent static:DictionaryScalarFnValuesPushDownRule slot=0 parent=vortex.binary(bool, len=4096) child=vortex.dict(i16, len=4096) -> vortex.dict(bool, len=4096)
      done output=vortex.dict(bool, len=4096)
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.binary(bool, len=4096) child=vortex.decimal_byte_parts(decimal(15,2), len=4096) -> vortex.dict(bool, len=4096)
      iter 1 current=vortex.dict(bool, len=4096) builder_active=false
        ExecuteSlot slot=0 parent=vortex.dict(bool, len=4096) child=fastlanes.bitpacked(u8, len=4096)
      iter 2 current=fastlanes.bitpacked(u8, len=4096) stack_parent=vortex.dict(bool, len=4096) slot=0 builder_active=false
        Done array=vortex.primitive(u8, len=4096)
      iter 3 current=vortex.primitive(u8, len=4096) stack_parent=vortex.dict(bool, len=4096) slot=0 builder_active=false
        pop_frame slot=0 output=vortex.dict(bool, len=4096)
      iter 4 current=vortex.dict(bool, len=4096) builder_active=false
        ExecuteSlot slot=1 parent=vortex.dict(bool, len=4096) child=vortex.binary(bool, len=50)
      iter 5 current=vortex.binary(bool, len=50) stack_parent=vortex.dict(bool, len=4096) slot=1 builder_active=false
    execute_until target=AnyCanonical root=vortex.sequence(i16, len=50)
      iter 0 current=vortex.sequence(i16, len=50) builder_active=false
        Done array=vortex.primitive(i16, len=50)
      iter 1 current=vortex.primitive(i16, len=50) builder_active=false
      return output=vortex.primitive(i16, len=50)
        Done array=vortex.bool(bool, len=50)
      iter 6 current=vortex.bool(bool, len=50) stack_parent=vortex.dict(bool, len=4096) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.dict(bool, len=4096)
      iter 7 current=vortex.dict(bool, len=4096) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(bool, len=4096) child=vortex.bool(bool, len=50) -> vortex.bool(bool, len=4096)
      iter 8 current=vortex.bool(bool, len=4096) builder_active=false
      return output=vortex.bool(bool, len=4096)
    ");

    Ok(())
}

/// Q12-style predicate over the shipmode column: `l_shipmode = 'AIR'`.
///
/// The column compresses to `dict -> {bitpacked codes, fsst values}`.
fn shipmode_predicate(column: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
    Binary.try_new_array(
        len,
        Operator::Eq,
        [
            column,
            ConstantArray::new(Scalar::from("AIR"), len).into_array(),
        ],
    )
}

#[test]
fn trace_scan_compare_on_compressed_shipmode() -> VortexResult<()> {
    let compressed = compressed_lineitem()?;
    let len = compressed.len();

    let lazy = shipmode_predicate(field(&compressed, "l_shipmode")?, len)?;
    let expected = shipmode_predicate(field(&lineitem()?, "l_shipmode")?, len)?;
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    insta::assert_snapshot!(optimized.trace.to_string(), @"
    optimize root=vortex.binary(bool, len=4096) session=false
      reduce_parent static:DictionaryScalarFnValuesPushDownRule slot=0 parent=vortex.binary(bool, len=4096) child=vortex.dict(utf8, len=4096) -> vortex.dict(bool, len=4096)
      done output=vortex.dict(bool, len=4096)
    ");
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.dict(bool, len=4096)
      iter 0 current=vortex.dict(bool, len=4096) builder_active=false
        ExecuteSlot slot=0 parent=vortex.dict(bool, len=4096) child=fastlanes.bitpacked(u8, len=4096)
      iter 1 current=fastlanes.bitpacked(u8, len=4096) stack_parent=vortex.dict(bool, len=4096) slot=0 builder_active=false
        Done array=vortex.primitive(u8, len=4096)
      iter 2 current=vortex.primitive(u8, len=4096) stack_parent=vortex.dict(bool, len=4096) slot=0 builder_active=false
        pop_frame slot=0 output=vortex.dict(bool, len=4096)
      iter 3 current=vortex.dict(bool, len=4096) builder_active=false
        ExecuteSlot slot=1 parent=vortex.dict(bool, len=4096) child=vortex.binary(bool, len=7)
      iter 4 current=vortex.binary(bool, len=7) stack_parent=vortex.dict(bool, len=4096) slot=1 builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.binary(bool, len=7) child=vortex.fsst(utf8, len=7) -> vortex.binary(bool, len=7)
      iter 5 current=vortex.binary(bool, len=7) stack_parent=vortex.dict(bool, len=4096) slot=1 builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.binary(bool, len=7) child=vortex.varbin(binary, len=7) -> vortex.bool(bool, len=7)
      iter 6 current=vortex.bool(bool, len=7) stack_parent=vortex.dict(bool, len=4096) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.dict(bool, len=4096)
      iter 7 current=vortex.dict(bool, len=4096) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(bool, len=4096) child=vortex.bool(bool, len=7) -> vortex.bool(bool, len=4096)
      iter 8 current=vortex.bool(bool, len=4096) builder_active=false
      return output=vortex.bool(bool, len=4096)
    ");

    Ok(())
}

/// Q13-style predicate over the comment column: `l_comment LIKE '%special%'`.
///
/// The column compresses to `fsst -> bitpacked lengths/offsets`.
fn comment_predicate(column: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
    Like.try_new_array(
        len,
        LikeOptions {
            negated: false,
            case_insensitive: false,
        },
        [
            column,
            ConstantArray::new(Scalar::from("%special%"), len).into_array(),
        ],
    )
}

#[test]
fn trace_scan_like_on_compressed_comment() -> VortexResult<()> {
    let compressed = compressed_lineitem()?;
    let len = compressed.len();

    let lazy = comment_predicate(field(&compressed, "l_comment")?, len)?;
    let expected = comment_predicate(field(&lineitem()?, "l_comment")?, len)?;
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    // No reduce rule rewrites a like over FSST; the FSST like kernel compiles the pattern and
    // matches in compressed space at execution time.
    insta::assert_snapshot!(optimized.trace.to_string(), @"");

    // The FSST like kernel materialises its offsets child
    // (`codes.offsets().execute::<PrimitiveArray>`), so no extra `execute_until` appears here only
    // because `EliasFanoScheme` declines under an FSST ancestor for exactly that reason. If that
    // exclusion is ever dropped, this trace grows an Elias-Fano decode inside the kernel.
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.like(bool, len=4096)
      iter 0 current=vortex.like(bool, len=4096) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.like(bool, len=4096) child=vortex.fsst(utf8, len=4096) -> vortex.bool(bool, len=4096)
      iter 1 current=vortex.bool(bool, len=4096) builder_active=false
      return output=vortex.bool(bool, len=4096)
    ");

    Ok(())
}

/// A narrowed projection of lineitem with one column per interesting compressed encoding:
/// decimal_byte_parts, ext-over-FoR dates, and dict-of-FSST strings.
fn project(table: &ArrayRef) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("l_quantity", field(table, "l_quantity")?),
        ("l_shipdate", field(table, "l_shipdate")?),
        ("l_shipmode", field(table, "l_shipmode")?),
    ])?
    .into_array())
}

#[test]
fn trace_scan_filter_on_compressed_table() -> VortexResult<()> {
    let compressed = project(&compressed_lineitem()?)?;
    let len = compressed.len();
    let mask = Mask::from_iter((0..len).map(|i| i % 97 == 0));

    let lazy = FilterArray::try_new(compressed, mask.clone())?.into_array();
    let expected = FilterArray::try_new(project(&lineitem()?)?, mask)?.into_array();
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    insta::assert_snapshot!(optimized.trace.to_string(), @"
    optimize root=vortex.filter({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43) session=false
      optimize root=vortex.filter(decimal(15,2), len=43) session=false
        optimize root=vortex.filter(i16, len=43) session=false
          reduce_parent static:FilterReduceAdaptor(Dict) slot=0 parent=vortex.filter(i16, len=43) child=vortex.dict(i16, len=4096) -> vortex.dict(i16, len=43)
          done output=vortex.dict(i16, len=43)
        reduce_parent static:DecimalBytePartsFilterPushDownRule slot=0 parent=vortex.filter(decimal(15,2), len=43) child=vortex.decimal_byte_parts(decimal(15,2), len=4096) -> vortex.decimal_byte_parts(decimal(15,2), len=43)
        done output=vortex.decimal_byte_parts(decimal(15,2), len=43)
      optimize root=vortex.filter(vortex.date[days](i32), len=43) session=false
        optimize root=vortex.filter(i32, len=43) session=false
          reduce_parent static:FoRFilterPushDownRule slot=0 parent=vortex.filter(i32, len=43) child=fastlanes.for(i32, len=4096) -> fastlanes.for(i32, len=43)
          done output=fastlanes.for(i32, len=43)
        reduce_parent static:ExtensionFilterPushDownRule slot=0 parent=vortex.filter(vortex.date[days](i32), len=43) child=vortex.ext(vortex.date[days](i32), len=4096) -> vortex.ext(vortex.date[days](i32), len=43)
        done output=vortex.ext(vortex.date[days](i32), len=43)
      optimize root=vortex.filter(utf8, len=43) session=false
        reduce_parent static:FilterReduceAdaptor(Dict) slot=0 parent=vortex.filter(utf8, len=43) child=vortex.dict(utf8, len=4096) -> vortex.dict(utf8, len=43)
        done output=vortex.dict(utf8, len=43)
      reduce FilterStructRule: vortex.filter({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43) -> vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43)
      done output=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43)
    ");
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43)
      iter 0 current=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43) builder_active=false
      return output=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=43)
    ");

    Ok(())
}

#[test]
fn trace_scan_take_on_compressed_table() -> VortexResult<()> {
    let compressed = project(&compressed_lineitem()?)?;
    let len = compressed.len() as u64;
    // A take is expressed as a `DictArray` whose codes are the take indices.
    let indices = PrimitiveArray::from_iter((0..64u64).map(|i| (i * 941) % len)).into_array();

    let lazy = DictArray::try_new(indices.clone(), compressed)?.into_array();
    let expected = DictArray::try_new(indices, project(&lineitem()?)?)?.into_array();
    let [optimized, executed] = optimize_then_execute(lazy, &expected)?;

    insta::assert_snapshot!(optimized.trace.to_string(), @"
    optimize root=vortex.dict({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64) session=false
      optimize root=vortex.dict(vortex.date[days](i32), len=64) session=false
        reduce_parent static:TakeReduceAdaptor(Extension) slot=1 parent=vortex.dict(vortex.date[days](i32), len=64) child=vortex.ext(vortex.date[days](i32), len=4096) -> vortex.ext(vortex.date[days](i32), len=64)
        done output=vortex.ext(vortex.date[days](i32), len=64)
      reduce_parent static:TakeReduceAdaptor(Struct) slot=1 parent=vortex.dict({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64) child=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=4096) -> vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64)
      done output=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64)
    ");
    insta::assert_snapshot!(executed.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64)
      iter 0 current=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64) builder_active=false
      return output=vortex.struct({l_quantity=decimal(15,2), l_shipdate=vortex.date[days](i32), l_shipmode=utf8}, len=64)
    ");

    Ok(())
}
