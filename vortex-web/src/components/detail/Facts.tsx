// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { ReactNode } from 'react';

/** The design system's key and value list (vx-facts): each key a label, each value in ink. */
export function Facts({
  items,
  className,
}: {
  items: [key: string, value: ReactNode, title?: string][];
  className?: string;
}) {
  return (
    <dl className={`vx-facts${className ? ` ${className}` : ''}`}>
      {items.map(([key, value, title]) => (
        <div key={key}>
          <dt className="vx-label-sm">{key}</dt>
          <dd {...(title ? { title } : {})}>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
