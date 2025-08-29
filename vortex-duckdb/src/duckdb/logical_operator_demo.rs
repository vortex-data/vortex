//! Demonstration of the new to_string methods for LogicalGet and LogicalProjection

use vortex::error::VortexResult;

use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorClass};

/// Demonstrate the to_string methods for specific operator types
pub fn demonstrate_operator_to_string(op: &LogicalOperator) -> VortexResult<()> {
    println!("=== LogicalOperator to_string Demo ===");

    // First show the general operator string
    println!("General operator: {}", op.to_debug_string()?);

    // Now show specialized to_string methods based on operator type
    match op.as_class() {
        Some(LogicalOperatorClass::Get(get_op)) => {
            println!("\n🔍 Detailed LogicalGet information:");
            println!("{}", get_op.to_string()?);
        }
        Some(LogicalOperatorClass::Projection(proj_op)) => {
            println!("\n📊 Detailed LogicalProjection information:");
            println!("{}", proj_op.to_string()?);
        }
        None => {
            println!("\n⚠️  This operator type doesn't have a specialized to_string method yet");
        }
    }

    Ok(())
}
