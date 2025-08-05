// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standardized error message utilities for consistent error reporting.

/// Creates a standardized cast error message.
/// 
/// Format: "Cannot cast {source_type} to {target_type}: {reason}"
#[macro_export]
macro_rules! cast_error {
    ($source:expr, $target:expr) => {
        vortex_err!("Cannot cast {} to {}", $source, $target)
    };
    
    ($source:expr, $target:expr, $reason:expr) => {
        vortex_err!("Cannot cast {} to {}: {}", $source, $target, $reason)
    };
}

/// Creates a standardized cast bail (immediate return) with error.
/// 
/// Format: "Cannot cast {source_type} to {target_type}: {reason}"
#[macro_export]
macro_rules! cast_bail {
    ($source:expr, $target:expr) => {
        vortex_bail!("Cannot cast {} to {}", $source, $target)
    };
    
    ($source:expr, $target:expr, $reason:expr) => {
        vortex_bail!("Cannot cast {} to {}: {}", $source, $target, $reason)
    };
}

#[cfg(test)]
mod tests {
    use vortex_error::{vortex_bail, vortex_err};
    
    #[test]
    fn test_error_macros() {
        // Test that the macros compile and produce expected format
        let err = cast_error!("i32", "bool");
        assert!(err.to_string().contains("Cannot cast i32 to bool"));
        
        let err_with_reason = cast_error!("decimal", "string", "unsupported conversion");
        assert!(err_with_reason.to_string().contains("Cannot cast decimal to string: unsupported conversion"));
    }
    
    #[test]
    fn test_bail_macro() {
        fn test_cast() -> vortex_error::VortexResult<()> {
            cast_bail!("bool", "struct", "incompatible types");
        }
        
        let result = test_cast();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot cast bool to struct: incompatible types"));
    }
}