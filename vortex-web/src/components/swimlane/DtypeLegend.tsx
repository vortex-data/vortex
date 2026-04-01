// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { DTYPE_CATEGORIES, DTYPE_COLORS } from './styles';

export function DtypeLegend() {
  return (
    <div className="flex gap-3.5 px-3 py-2 border-t border-vortex-grey-light/40 dark:border-white/[0.06] bg-vortex-white dark:bg-vortex-black text-[11px] text-vortex-grey-dark">
      {DTYPE_CATEGORIES.map((cat) => (
        <div key={cat} className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded" style={{ backgroundColor: DTYPE_COLORS[cat] }} />
          {cat}
        </div>
      ))}
    </div>
  );
}
