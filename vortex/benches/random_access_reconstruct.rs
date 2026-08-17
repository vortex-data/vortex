// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-memory reconstruction benchmarks for random-access feature-vector results.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::PType;
use vortex::array::patches::Patches;
use vortex::array::validity::Validity;
use vortex::arrow::ArrowSessionExt;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::ALPRDArrayExt;
use vortex::encodings::alp::ALPRDArrayOwnedExt;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

const NUM_CHUNKS: usize = 100;
const LIST_SIZE: usize = 1024;
const SEGMENT_ROWS: usize = 64;
const SELECTED_ROW: usize = 31;

#[derive(Clone)]
struct ResidentALPRDPage {
    left: BufferHandle,
    left_ptype: PType,
    left_bit_width: u8,
    right: BufferHandle,
    right_ptype: PType,
    right_bit_width: u8,
    dictionary: vortex::buffer::Buffer<u16>,
    dtype: vortex::dtype::DType,
    patch_indices: BufferHandle,
    patch_indices_ptype: PType,
    patch_values: BufferHandle,
    patch_values_ptype: PType,
    patch_count: usize,
    patch_offset: usize,
}

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn feature_values(chunk: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..LIST_SIZE).map(|index| {
        let mixed = (chunk * LIST_SIZE + index).wrapping_mul(2_654_435_761);
        (mixed % 100_003) as f32 / 100_003.0
    }))
}

fn segment_values() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..SEGMENT_ROWS * LIST_SIZE).map(|index| {
        let mixed = index.wrapping_mul(2_654_435_761);
        (mixed % 100_003) as f32 / 100_003.0
    }))
}

fn resident_alprd_page() -> ResidentALPRDPage {
    let values = segment_values();
    let encoded = RDEncoder::new(values.as_slice::<f32>()).encode(values.as_view());
    let patches = encoded.left_parts_patches().unwrap();
    let dictionary = encoded.left_parts_dictionary().clone();
    let right_bit_width = encoded.right_bit_width();
    let dtype = encoded.dtype().clone();
    let patch_offset = patches.offset();
    let mut ctx = SESSION.create_execution_ctx();
    let patch_indices = patches
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let patch_values = patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let patch_count = patch_indices.len();
    let patch_indices_ptype = patch_indices.ptype();
    let patch_values_ptype = patch_values.ptype();
    let patch_indices = patch_indices.buffer_handle().clone();
    let patch_values = patch_values.buffer_handle().clone();

    let parts = encoded.into_data_parts();
    let element_range = SELECTED_ROW * LIST_SIZE..(SELECTED_ROW + 1) * LIST_SIZE;
    let left_page = parts.left_parts.slice(element_range.clone()).unwrap();
    let right_page = parts.right_parts.slice(element_range).unwrap();
    let left_primitive = left_page
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let right_primitive = right_page
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let left = left_page.as_opt::<BitPacked>().unwrap();
    let right = right_page.as_opt::<BitPacked>().unwrap();
    assert_eq!(left.offset(), 0);
    assert_eq!(right.offset(), 0);

    ResidentALPRDPage {
        left: left.packed().clone(),
        left_ptype: left_primitive.ptype(),
        left_bit_width: left.bit_width(),
        right: right.packed().clone(),
        right_ptype: right_primitive.ptype(),
        right_bit_width,
        dictionary,
        dtype,
        patch_indices,
        patch_indices_ptype,
        patch_values,
        patch_values_ptype,
        patch_count,
        patch_offset,
    }
}

fn resident_pages() -> Vec<ResidentALPRDPage> {
    vec![resident_alprd_page(); NUM_CHUNKS]
}

fn reconstruct_resident_pages(
    pages: &[ResidentALPRDPage],
    ctx: &mut ExecutionCtx,
    eager_extract: bool,
) -> ArrayRef {
    let names: FieldNames = ["id", "embedding"].into_iter().collect();
    let chunks = pages
        .iter()
        .enumerate()
        .map(|(index, page)| {
            let patch_indices = PrimitiveArray::from_buffer_handle(
                page.patch_indices.clone(),
                page.patch_indices_ptype,
                Validity::NonNullable,
            )
            .into_array()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
            .into_array();
            let patch_values = PrimitiveArray::from_buffer_handle(
                page.patch_values.clone(),
                page.patch_values_ptype,
                Validity::NonNullable,
            )
            .into_array();
            let full_patches = Patches::new(
                SEGMENT_ROWS * LIST_SIZE,
                page.patch_offset,
                patch_indices,
                patch_values,
                None,
            )
            .unwrap();
            let element_start = SELECTED_ROW * LIST_SIZE;
            let patches = full_patches
                .slice(element_start..element_start + LIST_SIZE)
                .unwrap();
            let left = BitPacked::try_new(
                page.left.clone(),
                page.left_ptype,
                Validity::NonNullable,
                None,
                page.left_bit_width,
                LIST_SIZE,
                0,
            )
            .unwrap()
            .into_array();
            let right = BitPacked::try_new(
                page.right.clone(),
                page.right_ptype,
                Validity::NonNullable,
                None,
                page.right_bit_width,
                LIST_SIZE,
                0,
            )
            .unwrap()
            .into_array();
            let elements = ALPRD::try_new(
                page.dtype.clone(),
                left,
                page.dictionary.clone(),
                right,
                page.right_bit_width,
                patches,
            )
            .unwrap()
            .into_array();
            let elements = if eager_extract {
                elements
                    .execute::<PrimitiveArray>(ctx)
                    .unwrap()
                    .into_array()
            } else {
                elements
            };
            let embedding = FixedSizeListArray::try_new(
                elements,
                u32::try_from(LIST_SIZE).unwrap(),
                Validity::NonNullable,
                1,
            )
            .unwrap()
            .into_array();
            for child in embedding.depth_first_traversal() {
                child.statistics().clear_all();
            }
            let id = PrimitiveArray::from_iter([i64::try_from(index).unwrap()]).into_array();
            StructArray::try_new(names.clone(), [id, embedding], 1, Validity::NonNullable)
                .unwrap()
                .into_array()
        })
        .collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype)
        .unwrap()
        .into_array()
        .execute::<Canonical>(ctx)
        .unwrap()
        .into_array()
}

fn canonical_chunks() -> ChunkedArray {
    let list_size = u32::try_from(LIST_SIZE).unwrap();
    let chunks = (0..NUM_CHUNKS).map(|chunk| {
        FixedSizeListArray::new(
            feature_values(chunk).into_array(),
            list_size,
            Validity::NonNullable,
            1,
        )
        .into_array()
    });
    let chunks = chunks.collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap()
}

fn alprd_chunks() -> ChunkedArray {
    let list_size = u32::try_from(LIST_SIZE).unwrap();
    let chunks = (0..NUM_CHUNKS).map(|chunk| {
        let values = feature_values(chunk);
        let encoder = RDEncoder::new(values.as_slice::<f32>());
        let encoded = encoder.encode(values.as_view()).into_array();
        FixedSizeListArray::new(encoded, list_size, Validity::NonNullable, 1).into_array()
    });
    let chunks = chunks.collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap()
}

#[divan::bench]
fn concat_100_canonical_one_row_arrays(bencher: Bencher) {
    let chunked = canonical_chunks().into_array();
    bencher
        .with_inputs(|| (&chunked, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx).unwrap());
}

#[divan::bench]
fn container_concat_100_alprd_one_row_arrays(bencher: Bencher) {
    let chunked = alprd_chunks().into_array();
    bencher
        .with_inputs(|| (&chunked, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx).unwrap());
}

#[divan::bench]
fn rebuild_100_flat_arrays_from_resident_buffers(bencher: Bencher) {
    let pages = resident_pages();
    assert!(pages[0].patch_count > 0);
    bencher
        .with_inputs(|| (&pages, SESSION.create_execution_ctx()))
        .bench_refs(|(pages, ctx)| reconstruct_resident_pages(pages, ctx, false));
}

#[divan::bench]
fn extract_100_prebuilt_arrays_to_arrow(bencher: Bencher) {
    let pages = resident_pages();
    let array = reconstruct_resident_pages(&pages, &mut SESSION.create_execution_ctx(), false);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            SESSION
                .arrow()
                .execute_arrow((*array).clone(), None, ctx)
                .unwrap()
        });
}

#[divan::bench]
fn rebuild_and_extract_100_arrays_to_arrow(bencher: Bencher) {
    let pages = resident_pages();
    assert!(pages[0].patch_count > 0);
    bencher
        .with_inputs(|| (&pages, SESSION.create_execution_ctx()))
        .bench_refs(|(pages, ctx)| {
            let array = reconstruct_resident_pages(pages, ctx, false);
            SESSION.arrow().execute_arrow(array, None, ctx).unwrap()
        });
}

#[divan::bench]
fn rebuild_with_eager_extract_100_arrays_to_arrow(bencher: Bencher) {
    let pages = resident_pages();
    assert!(pages[0].patch_count > 0);
    bencher
        .with_inputs(|| (&pages, SESSION.create_execution_ctx()))
        .bench_refs(|(pages, ctx)| {
            let array = reconstruct_resident_pages(pages, ctx, true);
            SESSION.arrow().execute_arrow(array, None, ctx).unwrap()
        });
}

#[divan::bench]
fn rebuild_100_flat_arrays_with_eager_leaf_decode(bencher: Bencher) {
    let pages = resident_pages();
    assert!(pages[0].patch_count > 0);
    bencher
        .with_inputs(|| (&pages, SESSION.create_execution_ctx()))
        .bench_refs(|(pages, ctx)| reconstruct_resident_pages(pages, ctx, true));
}

#[divan::bench]
fn arrow_export_100_predecoded_arrays(bencher: Bencher) {
    let pages = resident_pages();
    let array = reconstruct_resident_pages(&pages, &mut SESSION.create_execution_ctx(), true);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            SESSION
                .arrow()
                .execute_arrow((*array).clone(), None, ctx)
                .unwrap()
        });
}
