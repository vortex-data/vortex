// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { NotFound } from './components/explorer/NotFound.tsx';
import { ThemeProvider } from './contexts/ThemeContext.tsx';

// The 404 page's entry (404.html). The deploy serves it, with a 404 status, for any address other
// than the Explorer's own.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <NotFound />
    </ThemeProvider>
  </StrictMode>,
);
