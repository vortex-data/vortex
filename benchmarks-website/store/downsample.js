import { LTTB } from "downsample";
import { MAX_POINTS } from "./constants.js";

function lttbIndices(seriesMap, target) {
  const keys = [...seriesMap.keys()];
  if (!keys.length) return [];

  const length = seriesMap.get(keys[0])?.length || 0;
  if (length <= target) return [...Array(length).keys()];

  const averages = Array(length);
  for (let i = 0; i < length; i++) {
    let sum = 0;
    let count = 0;

    for (const series of seriesMap.values()) {
      const value = series[i]?.value ?? series[i];
      if (value != null && !Number.isNaN(value)) {
        sum += value;
        count++;
      }
    }

    averages[i] = [i, count ? sum / count : 0];
  }

  const indices = LTTB(averages, target).map((point) => Math.round(point[0]));
  if (!indices.includes(0)) indices.unshift(0);
  if (!indices.includes(length - 1)) indices.push(length - 1);
  return indices.sort((a, b) => a - b);
}

export function downsample(data, factor) {
  const target = Math.ceil(data.commits.length / factor);
  if (target >= data.commits.length) return data;

  const indices = [...new Set(
    lttbIndices(data.series, target).filter(
      (index) =>
        Number.isInteger(index) &&
        index >= 0 &&
        index < data.commits.length,
    ),
  )].sort((a, b) => a - b);

  if (!indices.length) return data;

  const series = new Map();
  for (const [seriesName, values] of data.series) {
    series.set(
      seriesName,
      indices.map((index) => values[index]),
    );
  }

  return {
    ...data,
    commits: indices.map((index) => data.commits[index]),
    series,
  };
}

export function downsampleLevel(length) {
  if (length <= MAX_POINTS) return "1x";
  if (length <= MAX_POINTS * 2) return "2x";
  if (length <= MAX_POINTS * 4) return "4x";
  return "8x";
}
