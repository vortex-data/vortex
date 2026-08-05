// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for cosine similarity over tensor columns.
//!
//! The arms cover pairwise columns and both representations of a broadcast query vector. These
//! names are intended to remain stable across scalar-function implementation changes so CodSpeed
//! can compare them against `develop`.
//!
//! Rows are derived from a fixed element budget rather than fixed per arm, so widening a vector
//! trades rows for elements instead of multiplying the work. See [`ELEMENTS`].

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Total `f64` elements per operand, held constant across widths: the row count is
/// `ELEMENTS / width`. This budget is a quarter of the one the other tensor benches use, because
/// the constant arms recompute the broadcast vector's norm per row and cost roughly ten times the
/// column arms per element. It is what keeps every arm inside the 1 ms per-iteration limit from
/// `docs/developer-guide/benchmarking.md`, measured against CodSpeed's CPU simulation.
const ELEMENTS: usize = 2_048;

/// Widths chosen to separate the two costs, as in `l2_norm.rs`: the redundant norm pass is
/// `O(rows * width)`, one third of the closure's arithmetic, so wide tensors show the hoist
/// while a narrow one is dominated by per-row framework costs.
const WIDTHS: &[usize] = &[2, 32, 256];

/// `ELEMENTS / width` vectors of `width` `f64` elements, non-nullable. `seed` offsets the values so
/// the two sides of the column arm are not the same array.
fn vectors(width: usize, seed: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..ELEMENTS)
        .map(|i| (((i + seed) % 97) as f64) - 48.0)
        .collect();
    let storage = FixedSizeListArray::new(
        elements.into_array(),
        u32::try_from(width).unwrap(),
        Validity::NonNullable,
        ELEMENTS / width,
    )
    .into_array();
    Vector::try_new_vector_array(storage).unwrap()
}

/// One query vector of `width` `f64` elements broadcast to every row, as a [`ConstantArray`] over a
/// [`Vector`] extension scalar.
fn constant_vector(width: usize) -> ArrayRef {
    let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let children: Vec<Scalar> = (0..width)
        .map(|i| Scalar::primitive(((i % 97) as f64) - 48.0, Nullability::NonNullable))
        .collect();
    let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    let ext_scalar = Scalar::extension::<Vector>(EmptyMetadata, fsl_scalar);
    ConstantArray::new(ext_scalar, ELEMENTS / width).into_array()
}

fn bench_cosine(bencher: Bencher, lhs: ArrayRef, rhs: ArrayRef) {
    let session = vortex_array::array_session();
    bencher
        .with_inputs(|| {
            (
                CosineSimilarity::try_new_array(lhs.clone(), rhs.clone())
                    .unwrap()
                    .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// The control: both operands vary by row, so every norm must be computed in the row loop.
#[divan::bench(args = WIDTHS)]
fn column_x_column(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), vectors(width, 31));
}

/// The rhs is a broadcast query vector, whose norm is the same in every row.
#[divan::bench(args = WIDTHS)]
fn column_x_constant(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), constant_vector(width));
}

/// One query vector represented as an extension array over constant storage.
fn extension_constant_vector(width: usize) -> ArrayRef {
    let ext_dtype = vectors(width, 0).dtype().as_extension().clone();
    let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let children: Vec<Scalar> = (0..width)
        .map(|i| Scalar::primitive(((i % 97) as f64) - 48.0, Nullability::NonNullable))
        .collect();
    let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    ExtensionArray::new(
        ext_dtype,
        ConstantArray::new(fsl_scalar, ELEMENTS / width).into_array(),
    )
    .into_array()
}

/// The rhs is the same broadcast query represented as extension-wrapped constant storage.
#[divan::bench(args = WIDTHS)]
fn column_x_extension_constant(bencher: Bencher, width: usize) {
    bench_cosine(bencher, vectors(width, 0), extension_constant_vector(width));
}
