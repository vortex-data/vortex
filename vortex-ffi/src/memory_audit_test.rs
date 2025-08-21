#[cfg(test)]
mod tests {
    use vortex::arrays::StructArray;
    use vortex::buffer::Buffer;
    use vortex::{IntoArray, ArrayRef};
    
    use crate::array::vx_array;

    fn create_test_struct_array() -> ArrayRef {
        let nums: Buffer<i32> = (0..1000).collect();
        let floats: Buffer<f32> = (0..1000).map(|x| x as f32).collect();
        
        StructArray::try_from_iter([
            ("nums", nums.into_array()), 
            ("floats", floats.into_array())
        ])
        .unwrap()
        .into_array()
    }

    #[test]
    fn test_array_dtype_memory_safety() {
        println!("=== Testing array dtype memory safety ===");
        
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        
        // Get dtype reference using new_ref() - potentially dangerous
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };
        
        // Free the array - this is the dangerous scenario
        unsafe { crate::array::vx_array_free(vx_arr) };
        
        // Now try to use the dtype pointer - this could be use-after-free
        let variant = unsafe { crate::dtype::vx_dtype_get_variant(dtype_ptr) };
        println!("Variant after array freed: {:?}", variant);
        
        // This is potentially a use-after-free bug!
        // The dtype pointer may now point to freed memory
        
        println!("✅ Array dtype memory test completed (but may have use-after-free)");
    }

}