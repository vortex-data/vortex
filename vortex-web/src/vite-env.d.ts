// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** The repository's star count, formatted ("3.3k"), or '' when the build could not read it. */
  readonly VITE_GITHUB_STARS: string;
}
