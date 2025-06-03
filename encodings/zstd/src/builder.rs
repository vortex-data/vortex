use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::ZstdArray;

/// Builder for configuring zstd compression options
#[derive(Debug, Clone)]
pub struct ZstdBuilder {
    level: i32,
}

impl Default for ZstdBuilder {
    fn default() -> Self {
        Self { level: 3 }
    }
}

impl ZstdBuilder {
    /// Create a new ZstdBuilder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression level (1-22, where 1=fastest, 22=best compression)
    pub fn level(mut self, level: i32) -> Self {
        self.level = level;
        self
    }

    /// Build a ZstdArray from the given array using the configured settings
    pub fn build(self, array: ArrayRef) -> VortexResult<ZstdArray> {
        ZstdArray::try_from_array_with_level(array, self.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;

    #[test]
    fn test_builder_pattern() {
        let data = vec![1i32, 2, 3, 4, 5];
        let array = PrimitiveArray::from_vec(data, Validity::AllValid);
        
        // Using builder pattern
        let compressed = ZstdBuilder::new()
            .level(9)
            .build(array.into_array())
            .unwrap();
        
        assert_eq!(compressed.len(), 5);
    }

    #[test]
    fn test_default_builder() {
        let data = vec![1i32, 2, 3, 4, 5];
        let array = PrimitiveArray::from_vec(data, Validity::AllValid);
        
        // Using default settings
        let compressed = ZstdBuilder::default()
            .build(array.into_array())
            .unwrap();
        
        assert_eq!(compressed.len(), 5);
    }
}