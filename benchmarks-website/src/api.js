const API_BASE = '';

export async function fetchMetadata() {
  const response = await fetch(`${API_BASE}/api/metadata`);
  if (!response.ok) throw new Error(`Failed to fetch metadata: ${response.status}`);
  return response.json();
}

export async function fetchChartData(groupName, chartName, options = {}) {
  const { startTimestamp, endTimestamp, last, startIdx, endIdx } = options;
  let url = `${API_BASE}/api/data/${encodeURIComponent(groupName)}/${encodeURIComponent(chartName)}`;
  const params = new URLSearchParams();

  if (last) {
    params.set('last', last);
  } else if (startIdx !== undefined || endIdx !== undefined) {
    // Index-based range
    if (startIdx !== undefined) params.set('startIdx', startIdx);
    if (endIdx !== undefined) params.set('endIdx', endIdx);
  } else {
    // Timestamp-based range
    if (startTimestamp) {
      const ts = typeof startTimestamp === 'number'
        ? startTimestamp
        : new Date(startTimestamp).getTime();
      params.set('start', ts);
    }
    if (endTimestamp) {
      const ts = typeof endTimestamp === 'number'
        ? endTimestamp
        : new Date(endTimestamp).getTime();
      params.set('end', ts);
    }
  }

  if (params.toString()) url += '?' + params.toString();

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch chart data: ${response.status}`);
  return response.json();
}
