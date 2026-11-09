// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

pub use vortex_utils::tree::DepthContext as PlanTreeContext;
pub use vortex_utils::tree::IndentedFormatter as PlanIndentedFormatter;
use vortex_utils::tree::TreeDisplayAdapter;
pub use vortex_utils::tree::TreeDisplayExtractor as PlanTreeExtractor;
use vortex_utils::tree::write_indented_tree;

use super::PlanRef;

/// Adds the plan's display representation to a tree node's header.
pub struct PlanSummaryExtractor;

impl PlanSummaryExtractor {
    /// Writes a plan directly to `formatter`.
    pub fn write(plan: &PlanRef, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{plan}")
    }
}

impl PlanTreeExtractor<PlanRef, PlanTreeContext> for PlanSummaryExtractor {
    fn write_header(
        &self,
        plan: &PlanRef,
        _context: &PlanTreeContext,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, " ")?;
        Self::write(plan, formatter)
    }
}

/// Composable display builder for a physical plan tree.
///
/// Call `plan.tree_display()` for the default extractors. Use `plan.tree_display_builder()` to
/// start with only node and child names, then add extractors with [`Self::with`].
pub struct PlanTreeDisplay<'a> {
    plan: &'a PlanRef,
    extractors: Vec<Box<dyn PlanTreeExtractor<PlanRef, PlanTreeContext>>>,
}

impl<'a> PlanTreeDisplay<'a> {
    /// Creates a tree display for `plan` with no extractors.
    pub fn new(plan: &'a PlanRef) -> Self {
        Self {
            plan,
            extractors: Vec::new(),
        }
    }

    /// Creates a tree display using each plan's display representation.
    pub fn default_display(plan: &'a PlanRef) -> Self {
        Self::new(plan).with(PlanSummaryExtractor)
    }

    /// Adds an extractor to the display pipeline.
    pub fn with<E: PlanTreeExtractor<PlanRef, PlanTreeContext> + 'static>(
        mut self,
        extractor: E,
    ) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Adds a pre-boxed extractor to the display pipeline.
    pub fn with_boxed(
        mut self,
        extractor: Box<dyn PlanTreeExtractor<PlanRef, PlanTreeContext>>,
    ) -> Self {
        self.extractors.push(extractor);
        self
    }
}

impl TreeDisplayAdapter for PlanTreeDisplay<'_> {
    type Context = PlanTreeContext;
    type Node = PlanRef;

    fn write_node(
        &self,
        plan: &PlanRef,
        context: &PlanTreeContext,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        for extractor in &self.extractors {
            extractor.write_header(plan, context, formatter)?;
        }
        Ok(())
    }

    fn write_details(
        &self,
        plan: &PlanRef,
        context: &PlanTreeContext,
        formatter: &mut PlanIndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        for extractor in &self.extractors {
            extractor.write_details(plan, context, formatter)?;
        }
        Ok(())
    }

    fn visit_children(
        &self,
        plan: &PlanRef,
        visit: &mut dyn FnMut(&str, &PlanRef, bool) -> fmt::Result,
    ) -> fmt::Result {
        let mut present = plan.slots().present().peekable();
        while let Some((slot, lazy)) = present.next() {
            let child = lazy.get().map_err(|_| fmt::Error)?;
            visit(&plan.slot_name(slot), child, present.peek().is_none())?;
        }
        Ok(())
    }
}

impl fmt::Display for PlanTreeDisplay<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_indented_tree(
            self,
            "root",
            self.plan,
            &mut PlanTreeContext::default(),
            formatter,
        )
    }
}
