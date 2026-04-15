// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-chunk ingest transform.
//!
//! Bridges the parquet record-batch stream and the Vortex file writer:
//!
//! 1. Project the `emb` column out of each struct chunk.
//! 2. Rewrap the `emb` column as `Extension<Vector<f32, dim>>` via
//!    [`vortex_bench::conversions::list_to_vector_ext`].
//! 3. Cast the FSL element buffer from `f64` → `f32` if the source is `f64`. After this
//!    point all downstream code (compression, scan, recall) is f32-only.
//! 4. Optionally project the `scalar_labels` column through unchanged so future
//!    filtered-search benchmarks have it without re-ingest.
//! 5. Repackage as `Struct { emb: Vector<f32, dim>[, scalar_labels] }`.

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::extension::EmptyMetadata;
use vortex::array::validity::Validity;
use vortex::buffer::BufferMut;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_tensor::vector::Vector;

/// Configuration passed alongside each chunk so the transform can stay stateless.
#[derive(Debug, Clone, Copy)]
pub struct ChunkTransform {
    /// Source element ptype as declared by the dataset catalog. Used purely to decide
    /// whether the f64 → f32 cast is needed.
    pub src_ptype: PType,
    /// Whether to project the `scalar_labels` column through the output struct.
    pub include_scalar_labels: bool,
}

impl ChunkTransform {
    /// Apply the transform to a single struct chunk and return the rebuilt chunk.
    ///
    /// `chunk` must be a non-chunked `Struct` array carrying at least the `emb` column.
    /// The returned array is always a `Struct { emb: Vector<f32, dim>[, scalar_labels] }`.
    pub fn apply(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let struct_view = chunk.as_opt::<Struct>().with_context(|| {
            format!("ingest: expected struct chunk, got dtype {}", chunk.dtype())
        })?;

        let emb = struct_view
            .unmasked_field_by_name("emb")
            .context("ingest: chunk missing `emb` column")?
            .clone();
        let emb_ext: ExtensionArray = list_to_vector_ext(emb)?.execute(ctx)?;
        let emb_f32 = if self.src_ptype == PType::F64 {
            cast_vector_ext_to_f32(&emb_ext, ctx)?
        } else {
            emb_ext.into_array()
        };

        let mut fields: Vec<(&str, ArrayRef)> = Vec::with_capacity(2);
        fields.push(("emb", emb_f32));
        if self.include_scalar_labels {
            let labels = struct_view
                .unmasked_field_by_name("scalar_labels")
                .context("ingest: chunk missing `scalar_labels` column")?
                .clone();
            fields.push(("scalar_labels", labels));
        }

        Ok(StructArray::from_fields(&fields)?.into_array())
    }
}

/// Cast a `Vector<f64, dim>` extension array down to `Vector<f32, dim>`. The cast is lossy
/// — the bench operates entirely in f32 from this point on, matching the precision of
/// TurboQuant and the handrolled baseline.
fn cast_vector_ext_to_f32(ext: &ExtensionArray, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    if elements.ptype() != PType::F64 {
        bail!(
            "cast_vector_ext_to_f32: expected f64 elements, got {}",
            elements.ptype()
        );
    }
    let f64_slice = elements.as_slice::<f64>();
    let mut f32_buf = BufferMut::<f32>::with_capacity(f64_slice.len());
    for &v in f64_slice {
        // Lossy by design — we always operate in f32 from the prepare step onward.
        #[expect(clippy::cast_possible_truncation)]
        f32_buf.push(v as f32);
    }
    let f32_elements =
        PrimitiveArray::new::<f32>(f32_buf.freeze(), Validity::NonNullable).into_array();
    let dim = match fsl.dtype() {
        DType::FixedSizeList(_, dim, _) => *dim,
        other => bail!("cast_vector_ext_to_f32: expected FSL dtype, got {other}"),
    };
    let new_fsl = FixedSizeListArray::try_new(f32_elements, dim, Validity::NonNullable, fsl.len())?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, new_fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, new_fsl.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::List;
    use vortex::dtype::Nullability;
    use vortex::session::VortexSession;

    use super::*;

    fn list_chunk_f64(rows: &[&[f64]]) -> ArrayRef {
        let mut elements = BufferMut::<f64>::with_capacity(rows.iter().map(|r| r.len()).sum());
        let mut offsets = BufferMut::<i32>::with_capacity(rows.len() + 1);
        offsets.push(0);
        for row in rows {
            for &v in row.iter() {
                elements.push(v);
            }
            offsets.push(i32::try_from(elements.len()).unwrap());
        }
        let elements_array =
            PrimitiveArray::new::<f64>(elements.freeze(), Validity::NonNullable).into_array();
        let offsets_array =
            PrimitiveArray::new::<i32>(offsets.freeze(), Validity::NonNullable).into_array();
        vortex::array::Array::<List>::new(elements_array, offsets_array, Validity::NonNullable)
            .into_array()
    }

    #[test]
    fn f64_chunk_is_cast_to_f32() -> Result<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();

        let emb = list_chunk_f64(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let chunk = StructArray::from_fields(&[("emb", emb)])?.into_array();
        let transform = ChunkTransform {
            src_ptype: PType::F64,
            include_scalar_labels: false,
        };
        let out = transform.apply(chunk, &mut ctx)?;
        let out_struct = out.as_opt::<Struct>().expect("returns Struct");
        let out_emb = out_struct.unmasked_field_by_name("emb").unwrap().clone();
        let DType::Extension(ext) = out_emb.dtype() else {
            panic!("expected extension dtype, got {}", out_emb.dtype());
        };
        match ext.storage_dtype() {
            DType::FixedSizeList(elem, 3, Nullability::NonNullable) => {
                assert_eq!(
                    **elem,
                    DType::Primitive(PType::F32, Nullability::NonNullable)
                );
            }
            other => panic!("unexpected storage dtype {other}"),
        }
        Ok(())
    }

    #[test]
    fn f32_chunk_passes_through() -> Result<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();

        let mut elements = BufferMut::<f32>::with_capacity(6);
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            elements.push(v);
        }
        let mut offsets = BufferMut::<i32>::with_capacity(3);
        offsets.push(0);
        offsets.push(3);
        offsets.push(6);
        let emb = vortex::array::Array::<List>::new(
            PrimitiveArray::new::<f32>(elements.freeze(), Validity::NonNullable).into_array(),
            PrimitiveArray::new::<i32>(offsets.freeze(), Validity::NonNullable).into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let chunk = StructArray::from_fields(&[("emb", emb)])?.into_array();

        let transform = ChunkTransform {
            src_ptype: PType::F32,
            include_scalar_labels: false,
        };
        let out = transform.apply(chunk, &mut ctx)?;
        let out_struct = out.as_opt::<Struct>().expect("returns Struct");
        assert_eq!(out_struct.len(), 2);
        Ok(())
    }
}
