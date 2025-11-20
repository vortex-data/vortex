pub mod context;
pub mod optimizer;
pub mod rules;
#[cfg(test)]
mod tests;

pub use context::ArrayRuleContext;
pub use optimizer::ArrayOptimizer;
pub use rules::{AnyParent, ArrayParentMatcher, ArrayParentReduceRule, ArrayReduceRule};
