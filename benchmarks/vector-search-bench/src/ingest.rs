// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-chunk ingest transform.
//!
//! Bridges the parquet record-batch stream and the Vortex file writer:
//!
//! 1. Project the `emb` column out of each struct chunk.
//! 2. Rewrap the `emb` column as `Extension<Vector<f32, dim>>` via
//!    [`vortex_bench::vector_dataset::list_to_vector_ext`].
//! 3. Detect the FSL element ptype at runtime and cast `f64` -> `f32` when needed. Detection is
//!    from the arrow schema rather than a catalog declaration so upstream parquets whose actual
//!    precision disagrees with the catalog still ingest correctly. After this point all
//!    downstream code (compression, scan, recall) is f32-only.
//! 4. Optionally project the `scalar_labels` column through unchanged so future filtered-search
//!    benchmarks have it without re-ingest.
//! 5. Repackage as `Struct { id: i64, emb: Vector<f32, dim>, scalar_labels: ??? }`.

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
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
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDType;
use vortex_bench::vector_dataset::list_to_vector_ext;
use vortex_tensor::vector::AnyVector;
use vortex_tensor::vector::Vector;

/// Apply the transform to a single struct chunk and return the rebuilt chunk.
///
/// `chunk` must be a non-chunked `Struct { id: i64, emb: List<f32> }`, where all of the list
/// elements are
///
/// The returned array is always a `Struct { id: i64, emb: Vector<f32, dim> }`.
pub fn transform_chunk(chunk: ArrayRef, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    let struct_view = chunk
        .as_opt::<Struct>()
        .with_context(|| format!("ingest: expected struct chunk, got dtype {}", chunk.dtype()))?;

    let id = struct_view
        .unmasked_field_by_name("id")
        .context("ingest: chunk missing `id` column")?
        .clone();
    let emb = struct_view
        .unmasked_field_by_name("emb")
        .context("ingest: chunk missing `emb` column")?
        .clone();

    let emb_ext: ExtensionArray = list_to_vector_ext(emb)?.execute(ctx)?;

    // Detect the actual FSL element ptype from the extension storage dtype. The dataset catalog
    // cannot be trusted here: at least one upstream parquet (`sift-medium-5m`) ships f64
    // embeddings despite the catalog advertising f32.
    let element_ptype = {
        let storage_dtype = emb_ext.storage_array().dtype();
        match storage_dtype {
            DType::FixedSizeList(elem, ..) => match elem.as_ref() {
                DType::Primitive(ptype, _) => *ptype,
                other => bail!("ingest: expected primitive FSL element dtype, got {other}"),
            },
            other => bail!("ingest: expected FSL storage dtype, got {other}"),
        }
    };

    let f32_vector_array = match element_ptype {
        PType::F32 => emb_ext.into_array(),
        PType::F64 => convert_f64_to_f32_vectors(&emb_ext, ctx)?,
        other => bail!("ingest: unsupported emb element ptype {other}, expected f32 or f64"),
    };

    let fields = [("id", id), ("emb", f32_vector_array)];
    Ok(StructArray::from_fields(&fields)?.into_array())
}

/// Convert a `Vector<f64, dim>` extension array down to `Vector<f32, dim>`.
///
/// This conversion is lossy, but we are generally ok with this because most vector search
/// operations do not demand a high amount of precision.
fn convert_f64_to_f32_vectors(ext: &ExtensionArray, ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
    ensure!(ext.ext_dtype().is::<AnyVector>());

    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(ctx)?;
    let validity = fsl.validity()?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
    ensure!(elements.ptype() == PType::F64);

    let dim = match fsl.dtype() {
        DType::FixedSizeList(_, dim, _) => *dim,
        other => bail!("cast_vector_ext_to_f32: expected FSL dtype, got {other}"),
    };

    let f64_slice = elements.as_slice::<f64>();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "this is intentionally lossy"
    )]
    let f32_buf: Buffer<f32> = f64_slice
        .iter()
        .copied()
        .map(|double| double as f32)
        .collect();

    let f32_elements = PrimitiveArray::new::<f32>(f32_buf, Validity::NonNullable).into_array();
    let new_fsl = FixedSizeListArray::try_new(f32_elements, dim, validity, fsl.len())?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, new_fsl.dtype().clone())?.erased();

    Ok(ExtensionArray::new(ext_dtype, new_fsl.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::List;
    use vortex::buffer::BufferMut;
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

    fn id_array(ids: &[i64]) -> ArrayRef {
        PrimitiveArray::new::<i64>(
            BufferMut::from_iter(ids.iter().copied()).freeze(),
            Validity::NonNullable,
        )
        .into_array()
    }

    #[test]
    fn f64_chunk_is_cast_to_f32() -> Result<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();

        let emb = list_chunk_f64(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let chunk =
            StructArray::from_fields(&[("id", id_array(&[0, 1])), ("emb", emb)])?.into_array();
        let out = transform_chunk(chunk, &mut ctx)?;
        let out_struct = out
            .as_opt::<Struct>()
            .context("transform_chunk should return a Struct array")?;
        let out_emb = out_struct
            .unmasked_field_by_name("emb")
            .context("transform_chunk output should contain an emb field")?
            .clone();
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
        let chunk =
            StructArray::from_fields(&[("id", id_array(&[0, 1])), ("emb", emb)])?.into_array();

        let out = transform_chunk(chunk, &mut ctx)?;
        let out_struct = out.as_opt::<Struct>().expect("returns Struct");
        assert_eq!(out_struct.len(), 2);
        Ok(())
    }
}
