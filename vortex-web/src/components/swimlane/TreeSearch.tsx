// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useState, useCallback } from 'react';

interface TreeSearchProps {
  onSearch: (query: string) => void;
}

export function TreeSearch({ onSearch }: TreeSearchProps) {
  const [value, setValue] = useState('');

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const q = e.target.value;
      setValue(q);
      onSearch(q);
    },
    [onSearch],
  );

  const handleClear = useCallback(() => {
    setValue('');
    onSearch('');
  }, [onSearch]);

  return (
    <div className="flex items-center gap-1">
      <span className="text-[10px] text-vortex-grey-dark">&#x2315;</span>
      <input
        type="text"
        value={value}
        onChange={handleChange}
        placeholder="Filter…"
        className="flex-1 min-w-0 bg-transparent text-[11px] text-vortex-black dark:text-vortex-white outline-none placeholder:text-vortex-grey-dark/50"
      />
      {value && (
        <button
          onClick={handleClear}
          className="text-[10px] text-vortex-grey-dark hover:text-vortex-black dark:hover:text-vortex-white"
        >
          &#x2715;
        </button>
      )}
    </div>
  );
}
