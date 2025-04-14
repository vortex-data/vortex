use duckdb::core::FlatVector;
use vortex_mask::Mask;

pub fn write_validity_from_mask(mask: Mask, flat_vector: &mut FlatVector) {
    // Check that both the target vector is large enough and the mask too.
    // If we later allow vectors larger than 2k (against duckdb defaults), we can revisit this.
    assert!(mask.len() <= flat_vector.capacity());
    match mask {
        Mask::AllTrue(len) => {
            if let Some(slice) = flat_vector.validity_slice() {
                // This is only needed if the vector as previously allocated.
                slice[0..len].fill(u64::MAX)
            }
        }
        Mask::AllFalse(len) => {
            let slice = flat_vector.init_get_validity_slice();
            slice[0..len].fill(u64::MIN)
        }
        Mask::Values(arr) => {
            // TODO(joe): do this MUCH better, with a shifted u64 copy
            for (idx, v) in arr.boolean_buffer().iter().enumerate() {
                if !v {
                    flat_vector.set_null(idx);
                }
            }
        }
    }
}
