#[cfg(test)]
mod tests {
    use vortex::arrays::StructArray;
    use vortex::buffer::Buffer;
    use vortex::{IntoArray, ArrayRef};
    
    use crate::dtype::vx_dtype_struct_dtype;
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
    fn test_struct_introspection_simple() {
        println!("=== Testing simple struct field access ===");
        
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };
        
        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        let n_fields = unsafe { crate::struct_fields::vx_struct_fields_nfields(struct_fields_ptr) };
        assert_eq!(n_fields, 2);
        
        // Cleanup in reverse order - this is the safest order
        unsafe {
            crate::struct_fields::vx_struct_fields_free(struct_fields_ptr);
            crate::array::vx_array_free(vx_arr);
        }
        
        println!("✅ Simple struct introspection test passed");
    }

    #[test] 
    fn test_field_name_access() {
        println!("=== Testing field name access ===");
        
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };
        
        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        
        // Test field name access
        let field_name_ptr = unsafe { 
            crate::struct_fields::vx_struct_fields_field_name(struct_fields_ptr, 0) 
        };
        assert!(!field_name_ptr.is_null());
        
        let name_len = unsafe { crate::string::vx_string_len(field_name_ptr) };
        let name_ptr = unsafe { crate::string::vx_string_ptr(field_name_ptr) };
        let name_slice = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len) };
        let name_str = std::str::from_utf8(name_slice).unwrap();
        assert_eq!(name_str, "nums");
        
        // Cleanup in careful order
        unsafe {
            crate::string::vx_string_free(field_name_ptr);
            crate::struct_fields::vx_struct_fields_free(struct_fields_ptr);
            crate::array::vx_array_free(vx_arr);
        }
        
        println!("✅ Field name access test passed");
    }

    #[test]
    fn test_comprehensive_struct_introspection() {
        println!("=== Testing comprehensive struct introspection ===");
        
        let array = create_test_struct_array();
        let vx_arr = vx_array::new(array);
        let dtype_ptr = unsafe { crate::array::vx_array_dtype(vx_arr) };
        
        let struct_fields_ptr = unsafe { vx_dtype_struct_dtype(dtype_ptr) };
        let n_fields = unsafe { crate::struct_fields::vx_struct_fields_nfields(struct_fields_ptr) };
        assert_eq!(n_fields, 2);
        
        // Test both field names
        for i in 0..n_fields {
            let field_name_ptr = unsafe { 
                crate::struct_fields::vx_struct_fields_field_name(struct_fields_ptr, i as usize) 
            };
            assert!(!field_name_ptr.is_null());
            
            let name_len = unsafe { crate::string::vx_string_len(field_name_ptr) };
            let name_ptr = unsafe { crate::string::vx_string_ptr(field_name_ptr) };
            let name_slice = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len) };
            let name_str = std::str::from_utf8(name_slice).unwrap();
            
            let expected_name = if i == 0 { "nums" } else { "floats" };
            assert_eq!(name_str, expected_name);
            println!("Field {}: {}", i, name_str);
            
            unsafe {
                crate::string::vx_string_free(field_name_ptr);
            }
        }
        
        // Cleanup
        unsafe {
            crate::struct_fields::vx_struct_fields_free(struct_fields_ptr);
            crate::array::vx_array_free(vx_arr);
        }
        
        println!("✅ Comprehensive struct introspection test passed");
    }
}