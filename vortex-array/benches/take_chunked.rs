// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks [`ChunkedArray`] take routing across ordering, chunk count, duplicate density,
//! nullability, and value-shape configurations.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const SOURCE_LEN: usize = 1 << 20;
const NESTED_WIDTH: usize = 8;

#[derive(Clone, Copy)]
enum Pattern {
    Sorted,
    Shuffled,
    Repeated,
    Duplicate90,
    Duplicate99,
    SameChunk,
    AlternatingChunks,
    GroupedChunks,
}

impl Display for Pattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Sorted => "sorted",
            Self::Shuffled => "shuffled",
            Self::Repeated => "repeated",
            Self::Duplicate90 => "duplicate90",
            Self::Duplicate99 => "duplicate99",
            Self::SameChunk => "same_chunk",
            Self::AlternatingChunks => "alternating_chunks",
            Self::GroupedChunks => "grouped_chunks",
        })
    }
}

#[derive(Clone, Copy)]
enum ValueKind {
    Primitive,
    Struct8,
    FixedSizeList8,
}

impl Display for ValueKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Primitive => "primitive",
            Self::Struct8 => "struct8",
            Self::FixedSizeList8 => "fsl8",
        })
    }
}

#[derive(Clone, Copy)]
enum IndexValidity {
    NonNullable,
    AllValid,
    TenPercentNull,
}

impl Display for IndexValidity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NonNullable => "nonnull",
            Self::AllValid => "nullable_all_valid",
            Self::TenPercentNull => "nullable_10pct_null",
        })
    }
}

#[derive(Clone, Copy)]
struct Case {
    group: &'static str,
    pattern: Pattern,
    value_kind: ValueKind,
    validity: IndexValidity,
    chunks: usize,
    indices: usize,
}

impl Case {
    const fn new(group: &'static str, pattern: Pattern, chunks: usize, indices: usize) -> Self {
        Self {
            group,
            pattern,
            value_kind: ValueKind::Primitive,
            validity: IndexValidity::NonNullable,
            chunks,
            indices,
        }
    }

    const fn with_value_kind(mut self, value_kind: ValueKind) -> Self {
        self.value_kind = value_kind;
        self
    }

    const fn with_validity(mut self, validity: IndexValidity) -> Self {
        self.validity = validity;
        self
    }
}

impl Display for Case {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}/chunks={}/indices={}",
            self.group, self.pattern, self.value_kind, self.validity, self.chunks, self.indices
        )
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();

    for pattern in [Pattern::Sorted, Pattern::Shuffled] {
        for chunks in [16, 1_024] {
            for indices in [10_000, 100_000] {
                cases.push(Case::new("core", pattern, chunks, indices));
            }
        }
    }

    for chunks in [1_024, 16_384] {
        for indices in [1, 16, 256, 1_000] {
            cases.push(Case::new("small_m", Pattern::Shuffled, chunks, indices));
        }
    }

    for pattern in [
        Pattern::Repeated,
        Pattern::Duplicate90,
        Pattern::Duplicate99,
    ] {
        cases.push(Case::new("duplicates", pattern, 16, 100_000));
    }

    for pattern in [Pattern::Shuffled, Pattern::Duplicate99] {
        for value_kind in [
            ValueKind::Primitive,
            ValueKind::Struct8,
            ValueKind::FixedSizeList8,
        ] {
            cases.push(Case::new("value_shape", pattern, 16, 10_000).with_value_kind(value_kind));
        }
    }

    for chunks in [16, 1_024] {
        for validity in [IndexValidity::AllValid, IndexValidity::TenPercentNull] {
            cases.push(
                Case::new("nullable", Pattern::Sorted, chunks, 100_000).with_validity(validity),
            );
        }
    }

    for pattern in [
        Pattern::SameChunk,
        Pattern::AlternatingChunks,
        Pattern::GroupedChunks,
    ] {
        cases.push(Case::new("routing", pattern, 1_024, 100_000));
    }

    cases
}

fn primitive_values(start: usize, end: usize) -> ArrayRef {
    PrimitiveArray::from_iter((start..end).map(|value| u64::try_from(value).unwrap())).into_array()
}

fn chunk_values(value_kind: ValueKind, start: usize, end: usize) -> ArrayRef {
    match value_kind {
        ValueKind::Primitive => primitive_values(start, end),
        ValueKind::Struct8 => {
            let fields: Vec<_> = (0..NESTED_WIDTH)
                .map(|field| {
                    let field_offset = u64::try_from(field * SOURCE_LEN).unwrap();
                    PrimitiveArray::from_iter(
                        (start..end).map(|value| u64::try_from(value).unwrap() + field_offset),
                    )
                    .into_array()
                })
                .collect();
            StructArray::try_new(
                FieldNames::from(["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]),
                fields,
                end - start,
                Validity::NonNullable,
            )
            .unwrap()
            .into_array()
        }
        ValueKind::FixedSizeList8 => {
            let element_start = start * NESTED_WIDTH;
            let element_end = end * NESTED_WIDTH;
            FixedSizeListArray::new(
                primitive_values(element_start, element_end),
                u32::try_from(NESTED_WIDTH).unwrap(),
                Validity::NonNullable,
                end - start,
            )
            .into_array()
        }
    }
}

fn chunked_values(case: Case) -> ArrayRef {
    let chunk_len = SOURCE_LEN / case.chunks;
    let chunks = (0..case.chunks)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_len;
            let end = start + chunk_len;
            chunk_values(case.value_kind, start, end)
        })
        .collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

fn advance_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn index_values(case: Case) -> Vec<u64> {
    let mut state = 0x4d59_5df4_d0f3_3173u64;
    let mut indices = match case.pattern {
        Pattern::Sorted | Pattern::Shuffled => (0..case.indices)
            .map(|_| advance_random(&mut state) % u64::try_from(SOURCE_LEN).unwrap())
            .collect(),
        Pattern::Repeated => vec![u64::try_from(SOURCE_LEN / 2).unwrap(); case.indices],
        Pattern::Duplicate90 | Pattern::Duplicate99 => {
            let divisor = match case.pattern {
                Pattern::Duplicate90 => 10,
                Pattern::Duplicate99 => 100,
                _ => unreachable!(),
            };
            let unique_count = (case.indices / divisor).max(1);
            (0..case.indices)
                .map(|_| {
                    let slot = usize::try_from(
                        advance_random(&mut state) % u64::try_from(unique_count).unwrap(),
                    )
                    .unwrap();
                    u64::try_from(slot * SOURCE_LEN / unique_count).unwrap()
                })
                .collect()
        }
        Pattern::SameChunk => {
            let chunk_len = SOURCE_LEN / case.chunks;
            let chunk_start = (case.chunks / 2) * chunk_len;
            (0..case.indices)
                .map(|_| {
                    let local = usize::try_from(
                        advance_random(&mut state) % u64::try_from(chunk_len).unwrap(),
                    )
                    .unwrap();
                    u64::try_from(chunk_start + local).unwrap()
                })
                .collect()
        }
        Pattern::AlternatingChunks => {
            let chunk_len = SOURCE_LEN / case.chunks;
            (0..case.indices)
                .map(|position| {
                    let chunk_idx = if position.is_multiple_of(2) {
                        0
                    } else {
                        case.chunks - 1
                    };
                    let local = position / 2 % chunk_len;
                    u64::try_from(chunk_idx * chunk_len + local).unwrap()
                })
                .collect()
        }
        Pattern::GroupedChunks => {
            let chunk_len = SOURCE_LEN / case.chunks;
            (0..case.indices)
                .map(|position| {
                    let chunk_idx = position * case.chunks / case.indices;
                    let local = usize::try_from(
                        advance_random(&mut state) % u64::try_from(chunk_len).unwrap(),
                    )
                    .unwrap();
                    u64::try_from(chunk_idx * chunk_len + local).unwrap()
                })
                .collect()
        }
    };

    if matches!(case.pattern, Pattern::Sorted) {
        indices.sort_unstable();
    }
    indices
}

fn take_indices(case: Case) -> ArrayRef {
    let indices = index_values(case);
    match case.validity {
        IndexValidity::NonNullable => PrimitiveArray::from_iter(indices).into_array(),
        IndexValidity::AllValid => {
            PrimitiveArray::new(Buffer::from(indices), Validity::AllValid).into_array()
        }
        IndexValidity::TenPercentNull => PrimitiveArray::new(
            Buffer::from(indices),
            Validity::from_iter((0..case.indices).map(|index| !index.is_multiple_of(10))),
        )
        .into_array(),
    }
}

#[divan::bench(args = cases())]
fn take(bencher: Bencher, case: Case) {
    let values = chunked_values(case);
    let indices = take_indices(case);
    let session = array_session();

    bencher
        .counter(ItemsCount::new(case.indices))
        .with_inputs(|| (indices.clone(), session.create_execution_ctx()))
        .bench_refs(|(indices, ctx)| {
            values
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
