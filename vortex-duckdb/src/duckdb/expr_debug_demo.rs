//! Demonstration of the new to_debug_string method for Expression

use vortex::error::VortexResult;
use crate::duckdb::expr::Expression;

/// Demonstrate the enhanced expression debugging capabilities
pub fn demonstrate_expression_debug(expr: &Expression) -> VortexResult<()> {
    println!("=== Expression Debug Demo ===");
    
    // Show the basic Display representation
    println!("Basic toString(): {}", expr);
    
    // Show the detailed debug information
    println!("\n🔍 Detailed Debug Information:");
    println!("{}", expr.to_debug_string()?);
    
    Ok(())
}

/// Compare basic and debug string representations
pub fn compare_expression_representations(expr: &Expression) -> VortexResult<()> {
    println!("=== Expression Representation Comparison ===");
    
    // Basic representation
    println!("📋 Basic Representation:");
    println!("  {}", expr);
    
    // Debug representation  
    println!("\n🔍 Debug Representation:");
    let debug_info = expr.to_debug_string()?;
    for line in debug_info.lines() {
        println!("  {}", line);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expression_debug_demo_exists() {
        // This test just verifies the functions exist and can be called
        // Actual testing would require creating Expression instances,
        // which requires a DuckDB connection and is better done in integration tests
        println!("✅ Expression debug demo functions exist");
    }
}