// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Ambient declaration for global CSS side-effect imports such as `import
// './globals.css'` in the root layout. The tsconfig enables
// `noUncheckedSideEffectImports`, which requires every side-effect import to
// resolve to a typed module; Next.js compiles the CSS itself at build time, so
// this only needs to give the import a (contentless) module type. The more
// specific `*.module.css` typing Next ships still wins for CSS modules.
declare module '*.css';
