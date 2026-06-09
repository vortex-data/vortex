// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Pin the file-tracing root to this app. The repo has multiple lockfiles
  // (workspace root, the v2 benchmarks-website project, and this app), so
  // Next.js cannot infer the correct root on its own.
  outputFileTracingRoot: __dirname,
};

module.exports = nextConfig;
