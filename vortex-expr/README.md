<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# vortex-expr

User-authored, schema-independent expressions. A function is available to the authored API only when an `ExpressionFn` binding rule is registered. Execution-only scalar functions remain available through `vortex-array::expr::BoundExpression` without becoming part of the authored language.

The built-in registry covers the functions exposed by the authored constructors, including field access, literals, comparisons, casts, `LIKE`, `BETWEEN`, and selected struct and list operations. Rules can coerce literals where the authored API requires it; two differently typed literals still require an explicit cast. Binding an unregistered function returns an error.

The expression wire format supports built-in authored functions with existing option codecs. `case_when` and custom authored functions return a serialization error.
