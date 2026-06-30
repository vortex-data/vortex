// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { createContext, useContext } from 'react';
import type { VortexFileContextValue } from './VortexFileContext';

export const VortexFileContext = createContext<VortexFileContextValue | null>(null);

export function useVortexFile(): VortexFileContextValue {
  const ctx = useContext(VortexFileContext);
  if (!ctx) throw new Error('useVortexFile must be used within VortexFileProvider');
  return ctx;
}
