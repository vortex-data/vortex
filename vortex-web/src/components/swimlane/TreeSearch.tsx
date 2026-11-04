// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { useCallback, useId, useState } from 'react';

interface TreeSearchProps {
  onSearch: (query: string) => void;
}

/** A search field that filters the tree. Escape clears it, as in any search field. */
export function TreeSearch({ onSearch }: TreeSearchProps) {
  const [value, setValue] = useState('');
  const id = useId();

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setValue(e.target.value);
      onSearch(e.target.value);
    },
    [onSearch],
  );

  return (
    <>
      <label className="vx-sr-only" htmlFor={id}>
        Filter columns
      </label>
      <input
        id={id}
        className="vx-field is-sm"
        type="search"
        value={value}
        onChange={handleChange}
        placeholder="Column name"
        autoComplete="off"
        spellCheck={false}
      />
    </>
  );
}
