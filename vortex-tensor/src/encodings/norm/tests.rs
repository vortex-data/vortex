// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Extension;
use vortex::error::VortexResult;

use crate::encodings::norm::NormVectorArray;
use crate::utils::extension_list_size;
use crate::utils::extension_storage;
use crate::utils::extract_flat_elements;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::vector_array;

#[test]
fn encode_unit_vectors() -> VortexResult<()> {
    // Already unit-length vectors: norms should be 1.0 and vectors unchanged.
    let arr = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // norm = 1.0
            0.0, 1.0, 0.0, // norm = 1.0
        ],
    )?;

    let norm = NormVectorArray::compress(arr)?;
    let norms = norm.norms().to_canonical()?.into_primitive();
    assert_close(norms.as_slice::<f64>(), &[1.0, 1.0]);

    let vectors = norm.vector_array();
    let ext = vectors.dtype().as_extension_opt().unwrap();
    let list_size = extension_list_size(ext)?;
    let storage = extension_storage(vectors)?;
    let flat = extract_flat_elements(&storage, list_size)?;
    assert_close(flat.row::<f64>(0), &[1.0, 0.0, 0.0]);
    assert_close(flat.row::<f64>(1), &[0.0, 1.0, 0.0]);

    Ok(())
}

#[test]
fn encode_non_unit_vectors() -> VortexResult<()> {
    let arr = vector_array(
        2,
        &[
            3.0, 4.0, // norm = 5.0
            0.0, 0.0, // norm = 0.0 (zero vector)
        ],
    )?;

    let norm = NormVectorArray::compress(arr)?;
    let norms = norm.norms().to_canonical()?.into_primitive();
    assert_close(norms.as_slice::<f64>(), &[5.0, 0.0]);

    let vectors = norm.vector_array();
    let ext = vectors.dtype().as_extension_opt().unwrap();
    let list_size = extension_list_size(ext)?;
    let storage = extension_storage(vectors)?;
    let flat = extract_flat_elements(&storage, list_size)?;
    assert_close(flat.row::<f64>(0), &[3.0 / 5.0, 4.0 / 5.0]);
    assert_close(flat.row::<f64>(1), &[0.0, 0.0]);

    Ok(())
}

#[test]
fn execute_round_trip() -> VortexResult<()> {
    let original_elements = &[
        3.0, 4.0, // norm = 5.0
        6.0, 8.0, // norm = 10.0
    ];
    let arr = vector_array(2, original_elements)?;

    let norm = NormVectorArray::compress(arr)?;

    // Execute to reconstruct the original vectors.
    let mut ctx = vortex::array::LEGACY_SESSION.create_execution_ctx();
    let reconstructed = norm.decompress(&mut ctx)?;

    // The reconstructed array should be a Vector extension array.
    assert!(reconstructed.as_opt::<Extension>().is_some());

    let ext = reconstructed.dtype().as_extension_opt().unwrap();
    let list_size = extension_list_size(ext)?;
    let storage = extension_storage(&reconstructed)?;
    let flat = extract_flat_elements(&storage, list_size)?;
    assert_close(flat.row::<f64>(0), &[3.0, 4.0]);
    assert_close(flat.row::<f64>(1), &[6.0, 8.0]);

    Ok(())
}

#[test]
fn execute_round_trip_zero_vector() -> VortexResult<()> {
    let arr = vector_array(2, &[0.0, 0.0])?;

    let norm = NormVectorArray::compress(arr)?;

    let mut ctx = vortex::array::LEGACY_SESSION.create_execution_ctx();
    let reconstructed = norm.decompress(&mut ctx)?;

    let ext = reconstructed.dtype().as_extension_opt().unwrap();
    let list_size = extension_list_size(ext)?;
    let storage = extension_storage(&reconstructed)?;
    let flat = extract_flat_elements(&storage, list_size)?;
    // Zero vector should remain zero after round-trip.
    assert_close(flat.row::<f64>(0), &[0.0, 0.0]);

    Ok(())
}
