// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A layout plan captures an expression as it has been pushed into a layout tree. It is assembled
/// once per expression evaluation for the entire layout, vs a SplitPlan which is assembled once
/// per split of the layout.
pub trait LayoutPlan: 'static + Send + Sync {}
