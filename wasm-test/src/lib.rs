// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use wasm_bindgen::prelude::*;

// Helper macro for logging to browser console.
macro_rules! log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&format!($($t)*).into());
    }
}

#[wasm_bindgen]
pub struct VortexBenchmark {
    size: usize,
}

#[wasm_bindgen]
impl VortexBenchmark {
    /// Create a new benchmark instance.
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> VortexBenchmark {
        VortexBenchmark { size }
    }

    /// Test Vortex arrays to ensure the library is linked.
    pub fn test_vortex(&self) -> Result<(), JsValue> {
        use vortex::arrays::PrimitiveArray;
        use vortex::buffer::Buffer;
        use vortex::validity::Validity;

        // Create a simple integer array.
        let data: Vec<i32> = (0..self.size as i32).collect();
        let buffer = Buffer::from(data);
        let _array = PrimitiveArray::new(buffer, Validity::NonNullable);

        log!("Created Vortex PrimitiveArray with {} elements", self.size);

        // Test compute functions.
        use vortex::IntoArray;
        use vortex::arrays::ConstantArray;
        use vortex::compute::{Operator, compare, take};
        use vortex::scalar::Scalar;

        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = Buffer::from(data.clone());
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        // Create a constant array for comparison.
        let threshold_array = ConstantArray::new(Scalar::from(3i32), 5).into_array();
        let _comparison = compare(&array, &threshold_array, Operator::Gt)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Test take operation.
        let indices: Vec<u64> = vec![0, 2, 4];
        let indices_buffer = Buffer::from(indices);
        let indices_array = PrimitiveArray::new(indices_buffer, Validity::NonNullable).into_array();
        let _taken = take(&array, &indices_array).map_err(|e| JsValue::from_str(&e.to_string()))?;

        log!("Tested Vortex compute operations");

        // Test various encodings exist.
        use vortex::encodings;
        let _ = std::mem::size_of::<encodings::alp::ALPArray>();
        let _ = std::mem::size_of::<encodings::fastlanes::BitPackedArray>();
        let _ = std::mem::size_of::<encodings::runend::RunEndArray>();
        let _ = std::mem::size_of::<encodings::zigzag::ZigZagArray>();

        log!("Verified Vortex encodings are included");

        Ok(())
    }

    /// Test compression and decompression.
    pub fn test_compression(&self) -> Result<(), JsValue> {
        use vortex::Array;
        use vortex::arrays::PrimitiveArray;
        use vortex::buffer::buffer;
        use vortex::compressor::BtrBlocksCompressor;
        use vortex::validity::Validity;

        log!("Testing compression with BtrBlocksCompressor...");

        // Create an array with repeated values (good for compression).
        let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).to_array();
        let original_len = array.len();

        // Compress the array.
        let compressed = BtrBlocksCompressor::default()
            .compress(&array)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        log!(
            "Compressed array from {} to {} elements",
            original_len,
            compressed.len()
        );

        Ok(())
    }

    /// Test different array types.
    pub fn test_array_types(&self) -> Result<(), JsValue> {
        use vortex::IntoArray;
        use vortex::arrays::{ConstantArray, PrimitiveArray, StructArray};
        use vortex::buffer::Buffer;
        use vortex::scalar::Scalar;
        use vortex::validity::Validity;

        log!("Testing different array types...");

        // Test ConstantArray.
        let _const_array = ConstantArray::new(Scalar::from(42i32), 100);
        log!("Created ConstantArray with 100 elements of value 42");

        // Test StructArray.
        let field1 = PrimitiveArray::new(Buffer::from(vec![1i32, 2, 3]), Validity::NonNullable);
        let field2 = PrimitiveArray::new(Buffer::from(vec![4i32, 5, 6]), Validity::NonNullable);

        let _struct_array =
            StructArray::from_fields(&[("a", field1.into_array()), ("b", field2.into_array())])
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        log!("Created StructArray with 2 fields");

        // Test PrimitiveArray with different types.
        let _int_array =
            PrimitiveArray::new(Buffer::from(vec![1i64, 2, 3, 4]), Validity::NonNullable);
        let _float_array =
            PrimitiveArray::new(Buffer::from(vec![1.0f64, 2.0, 3.0]), Validity::NonNullable);
        log!("Created PrimitiveArrays with different numeric types");

        Ok(())
    }

    /// Test more compute operations.
    pub fn test_compute_ops(&self) -> Result<(), JsValue> {
        use vortex::IntoArray;
        use vortex::arrays::{ConstantArray, PrimitiveArray};
        use vortex::buffer::Buffer;
        use vortex::compute::{Operator, compare};
        use vortex::scalar::Scalar;
        use vortex::validity::Validity;

        log!("Testing additional compute operations...");

        let data: Vec<i32> = vec![10, 20, 30, 40, 50];
        let buffer = Buffer::from(data);
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        // Test comparison operations with a scalar converted to array.
        let scalar_array = ConstantArray::new(Scalar::from(25i32), 5).into_array();
        let _gt_result = compare(&array, &scalar_array, Operator::Gt)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        log!("Compared array elements > 25");

        // Test comparison with another array.
        let threshold_array = ConstantArray::new(Scalar::from(30i32), 5).into_array();
        let _comparison = compare(&array, &threshold_array, Operator::Gte)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        log!("Compared array elements >= 30");

        // Test equality comparison.
        let eq_array = ConstantArray::new(Scalar::from(30i32), 5).into_array();
        let _eq_result = compare(&array, &eq_array, Operator::Eq)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        log!("Compared array elements == 30");

        Ok(())
    }

    /// Get size info.
    pub fn get_size(&self) -> usize {
        self.size
    }
}

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    log!("Vortex WASM module initialized");
}

/// Get version information.
#[wasm_bindgen]
pub fn get_version() -> String {
    format!("vortex-wasm-test v{}", env!("CARGO_PKG_VERSION"))
}

/// A simple test function to verify WASM is working.
#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    use vortex::Array;
    use vortex::IntoArray;
    use vortex::arrays::{ConstantArray, PrimitiveArray, StructArray};
    use vortex::buffer::{Buffer, buffer};
    use vortex::compressor::BtrBlocksCompressor;
    use vortex::compute::{Operator, compare, take};
    use vortex::scalar::Scalar;
    use vortex::validity::Validity;

    #[wasm_bindgen_test]
    fn test_primitive_array() {
        let data: Vec<i32> = (0..1000).collect();
        let buffer = Buffer::from(data);
        let array = PrimitiveArray::new(buffer, Validity::NonNullable);
        assert_eq!(array.len(), 1000);
    }

    #[wasm_bindgen_test]
    fn test_compute_operations() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let buffer = Buffer::from(data);
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        // Test comparison.
        let threshold_array = ConstantArray::new(Scalar::from(3i32), 5).into_array();
        let comparison = compare(&array, &threshold_array, Operator::Gt).expect("compare failed");
        assert_eq!(comparison.len(), 5);

        // Test take.
        let indices: Vec<u64> = vec![0, 2, 4];
        let indices_buffer = Buffer::from(indices);
        let indices_array =
            PrimitiveArray::new(indices_buffer, Validity::NonNullable).into_array();
        let taken = take(&array, &indices_array).expect("take failed");
        assert_eq!(taken.len(), 3);
    }

    #[wasm_bindgen_test]
    fn test_encodings() {
        use vortex::encodings;

        // Verify encodings are linked by checking their sizes.
        let alp_size = std::mem::size_of::<encodings::alp::ALPArray>();
        let bitpacked_size = std::mem::size_of::<encodings::fastlanes::BitPackedArray>();
        let runend_size = std::mem::size_of::<encodings::runend::RunEndArray>();
        let zigzag_size = std::mem::size_of::<encodings::zigzag::ZigZagArray>();

        assert!(alp_size > 0);
        assert!(bitpacked_size > 0);
        assert!(runend_size > 0);
        assert!(zigzag_size > 0);
    }

    #[wasm_bindgen_test]
    fn test_compression() {
        let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).to_array();
        let original_len = array.len();

        let compressed = BtrBlocksCompressor::default()
            .compress(&array)
            .expect("compression failed");

        assert_eq!(compressed.len(), original_len);
    }

    #[wasm_bindgen_test]
    fn test_array_types() {
        // ConstantArray.
        let const_array = ConstantArray::new(Scalar::from(42i32), 100);
        assert_eq!(const_array.len(), 100);

        // StructArray.
        let field1 = PrimitiveArray::new(Buffer::from(vec![1i32, 2, 3]), Validity::NonNullable);
        let field2 = PrimitiveArray::new(Buffer::from(vec![4i32, 5, 6]), Validity::NonNullable);
        let struct_array =
            StructArray::from_fields(&[("a", field1.into_array()), ("b", field2.into_array())])
                .expect("StructArray creation failed");
        assert_eq!(struct_array.len(), 3);

        // Different numeric types.
        let int_array =
            PrimitiveArray::new(Buffer::from(vec![1i64, 2, 3, 4]), Validity::NonNullable);
        let float_array =
            PrimitiveArray::new(Buffer::from(vec![1.0f64, 2.0, 3.0]), Validity::NonNullable);
        assert_eq!(int_array.len(), 4);
        assert_eq!(float_array.len(), 3);
    }
}
