// ============================================================================
// Configuration
// ============================================================================

/**
 * Global configuration settings.
 */
const CONFIG = {
    wasmModulePath: "./pkg/vortex_wasm.js",
    githubRepo: "https://github.com/spiraldb/vortex",
    defaultVisibleCommits: 50,
    minWindowSize: 10,
    yAxisLabel: "Time (ms)",
};

/**
 * Benchmark group configurations. Each group contains multiple charts.
 */
const BENCHMARK_GROUPS = {
    "random-access": {
        title: "Random Access",
        charts: ["latency"],
        seriesNames: ["vortex", "parquet", "lance"],
        seriesColors: {
            "vortex": "#101010",
            "parquet": "#5DADE2",
            "lance": "#ef7f1d",
        },
    },
    "clickbench": {
        title: "ClickBench",
        charts: [
            "q00-nvme", "q01-nvme", "q02-nvme", "q03-nvme", "q04-nvme",
            "q05-nvme", "q06-nvme", "q07-nvme", "q08-nvme", "q09-nvme",
            "q10-nvme", "q11-nvme", "q12-nvme", "q13-nvme", "q14-nvme",
            "q15-nvme", "q16-nvme", "q17-nvme", "q18-nvme", "q19-nvme",
            "q20-nvme", "q21-nvme", "q22-nvme", "q23-nvme", "q24-nvme",
            "q25-nvme", "q26-nvme", "q27-nvme", "q28-nvme", "q29-nvme",
            "q30-nvme", "q31-nvme", "q32-nvme", "q33-nvme", "q34-nvme",
            "q35-nvme", "q36-nvme", "q37-nvme", "q38-nvme", "q39-nvme",
            "q40-nvme", "q41-nvme", "q42-nvme",
        ],
        seriesNames: [
            "duckdb:vortex-compact",
            "duckdb:vortex-file-compressed",
            "duckdb:parquet",
            "duckdb:duckdb",
            "datafusion:vortex-compact",
            "datafusion:vortex-file-compressed",
            "datafusion:parquet",
            "datafusion:lance",
        ],
        seriesColors: {
            "duckdb:vortex-compact": "#101010",
            "duckdb:vortex-file-compressed": "#6b7280",
            "duckdb:parquet": "#5DADE2",
            "duckdb:duckdb": "#f59e0b",
            "datafusion:vortex-compact": "#059669",
            "datafusion:vortex-file-compressed": "#10b981",
            "datafusion:parquet": "#8b5cf6",
            "datafusion:lance": "#ef7f1d",
        },
    },
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Updates the status display with a message and type.
 */
function setStatus(message, type = "loading") {
    const status = document.getElementById("status");
    const text = document.getElementById("status-text");
    status.className = `status ${type}`;
    text.textContent = message;

    const spinner = status.querySelector(".spinner");
    if (spinner) {
        spinner.style.display = type === "loading" ? "inline-block" : "none";
    }
}

/**
 * Formats time in human-readable format with appropriate units.
 */
function formatTime(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Creates a unique ID for chart elements.
 */
function makeId(groupId, chartName, suffix) {
    return `${groupId}-${chartName}-${suffix}`.replace(/[^a-zA-Z0-9-]/g, '-');
}

// ============================================================================
// Data Loading Functions
// ============================================================================

/**
 * Loads and initializes the WASM module with timeout.
 */
async function loadWasmModule() {
    setStatus("Loading WASM module...");
    console.log("[loadWasmModule] Starting import...");

    const timeout = (ms) => new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
    );

    try {
        // Race import against timeout.
        const wasm = await Promise.race([
            import(CONFIG.wasmModulePath),
            timeout(10000)
        ]);
        console.log("[loadWasmModule] Import complete, initializing...");

        // Race init against timeout.
        await Promise.race([
            wasm.default(),
            timeout(10000)
        ]);
        console.log("[loadWasmModule] Initialized:", wasm.get_version());

        return wasm;
    } catch (error) {
        console.error("[loadWasmModule] Failed:", error);
        throw error;
    }
}

/**
 * Loads benchmark summary from WASM (commits + group/chart metadata, no values).
 */
async function loadBenchmarkSummary(wasm) {
    setStatus("Loading benchmark summary...");
    const json = await wasm.load_benchmark_summary();
    const summary = JSON.parse(json);
    console.log(`Loaded summary with ${summary.commits.length} commits, ${Object.keys(summary.groups).length} groups`);
    return summary;
}

/**
 * Loads chart data for a specific group and chart.
 */
async function loadChartData(wasm, group, chart) {
    const json = await wasm.load_chart_data(group, chart);
    return JSON.parse(json);
}

// ============================================================================
// Data Processing Functions
// ============================================================================

/**
 * Processes chart data into chart-ready format.
 */
function processChartData(chartData, commits, groupConfig) {
    const alignedSeries = chartData.aligned_series;

    const processedCommits = commits.map((commit, index) => ({
        ...commit,
        id: commit.commit_id,
        url: `${CONFIG.githubRepo}/commit/${commit.commit_id}`,
        sortedIndex: index,
    }));

    const seriesData = new Map();
    for (const name of groupConfig.seriesNames) {
        const rawData = alignedSeries[name];
        if (rawData) {
            const msData = rawData.map(v => v !== null ? { value: Number(v) / 1_000_000 } : null);
            seriesData.set(name, msData);
        } else {
            seriesData.set(name, new Array(commits.length).fill(null));
        }
    }

    // Find the range of data.
    let firstDataIndex = commits.length;
    let lastDataIndex = -1;
    for (const data of seriesData.values()) {
        for (let i = 0; i < data.length; i++) {
            if (data[i] !== null) {
                firstDataIndex = Math.min(firstDataIndex, i);
                lastDataIndex = Math.max(lastDataIndex, i);
            }
        }
    }

    if (lastDataIndex < 0) {
        firstDataIndex = 0;
        lastDataIndex = commits.length - 1;
    }

    const startIndex = Math.max(0, firstDataIndex);
    const endIndex = lastDataIndex + 1;
    const chartCommits = processedCommits.slice(startIndex, endIndex);
    const slicedSeriesData = new Map();

    for (const [name, data] of seriesData.entries()) {
        slicedSeriesData.set(name, data.slice(startIndex, endIndex));
    }

    return { seriesData: slicedSeriesData, chartCommits };
}

/**
 * Calculates summary statistics for the latest data point.
 */
function calculateSummary(seriesData) {
    const latestResults = new Map();

    for (const [seriesName, data] of seriesData.entries()) {
        for (let i = data.length - 1; i >= 0; i--) {
            if (data[i] !== null) {
                latestResults.set(seriesName, data[i].value);
                break;
            }
        }
    }

    if (latestResults.size === 0) {
        return { results: [], fastestTime: 0 };
    }

    const fastestTime = Math.min(...latestResults.values());
    const sortedResults = Array.from(latestResults.entries())
        .sort((a, b) => a[1] - b[1]);

    return { results: sortedResults, fastestTime };
}

// ============================================================================
// Group Summary Data Storage
// ============================================================================

/**
 * Storage for per-chart latest values. Used for calculating group summaries.
 * Structure: groupId -> chartName -> Map<seriesName, latestValueMs>
 */
const groupChartData = new Map();

/**
 * Stores the latest value for each series in a chart.
 */
function storeChartData(groupId, chartName, seriesData) {
    if (!groupChartData.has(groupId)) {
        groupChartData.set(groupId, new Map());
    }
    const groupData = groupChartData.get(groupId);

    const latestValues = new Map();
    for (const [seriesName, data] of seriesData.entries()) {
        for (let i = data.length - 1; i >= 0; i--) {
            if (data[i] !== null) {
                latestValues.set(seriesName, data[i].value);
                break;
            }
        }
    }
    groupData.set(chartName, latestValues);
}

/**
 * Calculates group-level summary for multi-chart groups (like clickbench).
 *
 * For clickbench scoring:
 * - Score: Geometric mean of (query_time + 10ms) / (fastest_time + 10ms) across all queries
 * - Total: Sum of all query times
 */
function calculateGroupSummary(groupId, groupConfig) {
    const groupData = groupChartData.get(groupId);
    if (!groupData || groupData.size === 0) {
        return { results: [], isMultiChart: false };
    }

    const isMultiChart = groupConfig.charts.length > 1;

    if (!isMultiChart) {
        // Single chart group: use simple summary (latest values).
        const chartData = groupData.values().next().value;
        if (!chartData || chartData.size === 0) {
            return { results: [], isMultiChart: false };
        }

        const fastestTime = Math.min(...chartData.values());
        const sortedResults = Array.from(chartData.entries())
            .sort((a, b) => a[1] - b[1])
            .map(([name, time]) => ({
                name,
                time,
                ratio: time / fastestTime,
            }));

        return { results: sortedResults, isMultiChart: false };
    }

    // Multi-chart group: calculate geometric mean score and total time.
    const SHIFT_MS = 10; // Constant shift to avoid division issues with small values.
    const seriesStats = new Map(); // seriesName -> { ratios: [], total: 0 }

    // Initialize stats for all series.
    for (const seriesName of groupConfig.seriesNames) {
        seriesStats.set(seriesName, { ratios: [], total: 0 });
    }

    // Process each chart (query).
    for (const [chartName, chartValues] of groupData.entries()) {
        // Find fastest time for this query.
        let fastestTime = Infinity;
        for (const time of chartValues.values()) {
            if (time < fastestTime) fastestTime = time;
        }

        // Calculate ratio for each series.
        for (const seriesName of groupConfig.seriesNames) {
            const stats = seriesStats.get(seriesName);
            const time = chartValues.get(seriesName);

            if (time !== undefined) {
                const ratio = (time + SHIFT_MS) / (fastestTime + SHIFT_MS);
                stats.ratios.push(ratio);
                stats.total += time;
            }
        }
    }

    // Calculate geometric mean and build results.
    const results = [];
    for (const [seriesName, stats] of seriesStats.entries()) {
        if (stats.ratios.length === 0) continue;

        // Geometric mean = exp(mean(log(ratios))).
        const logSum = stats.ratios.reduce((sum, r) => sum + Math.log(r), 0);
        const geometricMean = Math.exp(logSum / stats.ratios.length);

        results.push({
            name: seriesName,
            score: geometricMean,
            total: stats.total,
            queryCount: stats.ratios.length,
        });
    }

    // Sort by score (lower is better).
    results.sort((a, b) => a.score - b.score);

    return { results, isMultiChart: true };
}

// ============================================================================
// HTML Generation Functions
// ============================================================================

/**
 * Creates HTML for a benchmark group container.
 */
function createGroupHTML(groupId, groupConfig) {
    return `
        <div id="${groupId}-group" class="benchmark-set collapsed">
            <div class="benchmark-header">
                <div class="title-wrapper">
                    <span class="collapse-icon">▼</span>
                    <h2 class="benchmark-title">${groupConfig.title}</h2>
                </div>
                <div class="benchmark-meta">
                    <span id="${groupId}-status">0/${groupConfig.charts.length} charts loaded</span>
                </div>
            </div>
            <div class="summary-section">
                <div id="${groupId}-summary" class="scores-list"></div>
                <p class="scores-explanation" id="${groupId}-summary-explanation"></p>
            </div>
            <div class="benchmark-graphs" id="${groupId}-charts">
                <!-- Charts will be rendered here -->
            </div>
        </div>
    `;
}

/**
 * Creates HTML for a single chart within a group.
 */
function createChartHTML(groupId, chartName) {
    const prefix = makeId(groupId, chartName, '');
    return `
        <div class="chart-section" id="${prefix}section">
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title">${chartName}</h3>
                    <div class="chart-controls">
                        <span id="${prefix}info" class="control-info-compact">Loading...</span>
                        <div class="zoom-controls">
                            <button id="${prefix}zoom-out-large" class="zoom-btn">−−</button>
                            <button id="${prefix}zoom-out-small" class="zoom-btn">−</button>
                            <button id="${prefix}zoom-in-small" class="zoom-btn">+</button>
                            <button id="${prefix}zoom-in-large" class="zoom-btn">++</button>
                        </div>
                    </div>
                </div>
                <div class="chart-wrapper">
                    <canvas id="${prefix}canvas"></canvas>
                </div>
                <div id="${prefix}tooltip" class="chartjs-tooltip"></div>
                <div class="x-axis-label">Commit</div>
                <div class="timeline-scrollbar-container" id="${prefix}scrollbar-container">
                    <div id="${prefix}scrollbar-content"></div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// Rendering Functions
// ============================================================================

/**
 * Renders the group-level summary.
 */
function renderGroupSummary(groupId, groupConfig) {
    const summaryList = document.getElementById(`${groupId}-summary`);
    const explanationEl = document.getElementById(`${groupId}-summary-explanation`);
    if (!summaryList) return;

    const summary = calculateGroupSummary(groupId, groupConfig);

    summaryList.innerHTML = "";

    if (summary.results.length === 0) {
        summaryList.innerHTML = '<div class="score-item">No data available</div>';
        if (explanationEl) explanationEl.textContent = "";
        return;
    }

    if (summary.isMultiChart) {
        // Multi-chart group (clickbench): show score and total.
        summary.results.forEach((result, index) => {
            const item = document.createElement("div");
            item.className = "score-item";
            item.innerHTML = `
                <span class="score-rank">#${index + 1}</span>
                <span class="score-series" style="color: ${groupConfig.seriesColors[result.name]}">${result.name}</span>
                <div class="score-metrics">
                    <span class="score-runtime">${result.score.toFixed(2)}x</span>
                    <span class="score-ratio">${formatTime(result.total)}</span>
                </div>
            `;
            summaryList.appendChild(item);
        });

        if (explanationEl) {
            explanationEl.textContent = "Score: geometric mean of query time ratio to fastest with 10ms constant shift | Total: sum of all query times (lower is better)";
        }
    } else {
        // Single-chart group: show time and ratio.
        summary.results.forEach((result, index) => {
            const item = document.createElement("div");
            item.className = "score-item";
            item.innerHTML = `
                <span class="score-rank">#${index + 1}</span>
                <span class="score-series" style="color: ${groupConfig.seriesColors[result.name]}">${result.name}</span>
                <div class="score-metrics">
                    <span class="score-runtime">${formatTime(result.time)}</span>
                    <span class="score-ratio">${result.ratio.toFixed(2)}x</span>
                </div>
            `;
            summaryList.appendChild(item);
        });

        if (explanationEl) {
            explanationEl.textContent = "Query time | Ratio to fastest (lower is better)";
        }
    }
}

/**
 * Resizes all charts in a group sequentially, yielding to the browser between each.
 * This ensures the first chart appears immediately rather than waiting for all 43.
 */
async function resizeGroupChartsSequentially(groupId) {
    for (const [prefix, chart] of chartInstances.entries()) {
        if (prefix.startsWith(groupId + '-')) {
            chart.resize();
            // Yield to browser to paint before next resize.
            await new Promise(r => requestAnimationFrame(r));
        }
    }
}

/**
 * Sets up collapsible benchmark sections.
 */
function setupCollapsibleBenchmarks() {
    document.querySelectorAll('.benchmark-header').forEach(header => {
        header.addEventListener('click', async () => {
            const benchmarkSet = header.closest('.benchmark-set');
            const wasCollapsed = benchmarkSet.classList.contains('collapsed');
            benchmarkSet.classList.toggle('collapsed');

            // If we just expanded, resize charts sequentially so first appears immediately.
            if (wasCollapsed) {
                const groupId = benchmarkSet.id.replace('-group', '');
                await resizeGroupChartsSequentially(groupId);
            }
        });
    });
}

/**
 * Creates Chart.js datasets from series data.
 */
function createDatasets(seriesData, groupConfig) {
    return groupConfig.seriesNames.map(name => {
        const data = seriesData.get(name);
        return {
            label: name,
            data: data ? data.map(d => d?.value ?? null) : [],
            borderColor: groupConfig.seriesColors[name],
            backgroundColor: groupConfig.seriesColors[name],
            borderWidth: 1.5,
            borderJoinStyle: 'round',
            pointRadius: 2,
            tension: 0,
            spanGaps: true,
        };
    });
}

/**
 * Creates empty Chart.js datasets (structure only, no data).
 */
function createEmptyDatasets(groupConfig) {
    return groupConfig.seriesNames.map(name => ({
        label: name,
        data: [],
        borderColor: groupConfig.seriesColors[name],
        backgroundColor: groupConfig.seriesColors[name],
        borderWidth: 1.5,
        borderJoinStyle: 'round',
        pointRadius: 2,
        tension: 0,
        spanGaps: true,
    }));
}

// ============================================================================
// Chart Configuration Functions
// ============================================================================

/**
 * Creates the vertical line plugin for Chart.js.
 */
function createVerticalLinePlugin() {
    return {
        id: 'verticalLine',
        afterDatasetsDraw(chart) {
            if (chart.tooltip?._active?.length) {
                const activePoint = chart.tooltip._active[0];
                const ctx = chart.ctx;
                const x = activePoint.element.x;
                const topY = chart.scales.y.top;
                const bottomY = chart.scales.y.bottom;

                ctx.save();
                ctx.beginPath();
                ctx.moveTo(x, topY);
                ctx.lineTo(x, bottomY);
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(89, 113, 253, 0.5)';
                ctx.stroke();
                ctx.restore();
            }
        }
    };
}

/**
 * Creates tooltip configuration for Chart.js.
 */
function createTooltipConfig(chartCommits, tooltipElementId) {
    return {
        enabled: false,
        external: (context) => renderExternalTooltip(context, tooltipElementId),
        callbacks: {
            footer: (tooltipItems) => getTooltipFooter(tooltipItems, chartCommits)
        }
    };
}

/**
 * Gets tooltip footer content with commit details.
 */
function getTooltipFooter(tooltipItems, chartCommits) {
    if (tooltipItems.length === 0) return [];
    const commit = chartCommits[tooltipItems[0].dataIndex];
    if (!commit) return [];

    const date = new Date(commit.timestamp * 1000).toLocaleDateString();

    return [
        commit.message.split("\n")[0].slice(0, 60),
        `${commit.author.name} - ${date}`
    ];
}

/**
 * Renders the external tooltip element.
 */
function renderExternalTooltip(context, tooltipElementId) {
    const tooltipEl = document.getElementById(tooltipElementId);
    if (!tooltipEl) return;

    const tooltipModel = context.tooltip;

    if (tooltipModel.opacity === 0) {
        tooltipEl.classList.remove('active');
        return;
    }

    if (tooltipModel.body) {
        tooltipEl.innerHTML = buildTooltipHTML(tooltipModel);
        positionTooltip(tooltipEl, context, tooltipModel);
    }
}

/**
 * Builds HTML content for the tooltip.
 */
function buildTooltipHTML(tooltipModel) {
    const titleLines = tooltipModel.title || [];
    const footerLines = tooltipModel.footer || [];

    let html = '<div class="chartjs-tooltip-body">';

    titleLines.forEach(title => {
        html += `<div style="font-weight: bold; margin-bottom: 4px;">${title}</div>`;
    });

    const sortedItems = [...tooltipModel.dataPoints].sort((a, b) => b.parsed.y - a.parsed.y);
    sortedItems.forEach((item) => {
        const color = item.dataset.borderColor;
        const value = item.formattedValue;
        const label = item.dataset.label;
        html += `
            <div class="chartjs-tooltip-item">
                <div class="chartjs-tooltip-color" style="background-color: ${color}"></div>
                <span>${label}: ${value}ms</span>
            </div>
        `;
    });

    if (footerLines.length > 0) {
        html += '<div class="chartjs-tooltip-footer">';
        footerLines.forEach(footer => {
            html += `<div>${footer}</div>`;
        });
        html += '</div>';
    }

    html += '</div>';
    return html;
}

/**
 * Positions the tooltip below the chart.
 */
function positionTooltip(tooltipEl, context, tooltipModel) {
    const canvas = context.chart.canvas;
    const canvasRect = canvas.getBoundingClientRect();

    tooltipEl.classList.add('active');
    tooltipEl.style.left = canvasRect.left + window.pageXOffset + tooltipModel.caretX + 'px';
    tooltipEl.style.top = canvasRect.bottom + window.pageYOffset + 10 + 'px';
    tooltipEl.style.transform = 'translateX(-50%)';
}

/**
 * Creates chart options configuration.
 */
function createChartOptions(chartCommits, tooltipElementId, groupId, groupConfig) {
    const legendConfig = {
        position: "top",
    };

    // Add group-linked legend click handler if groupId and groupConfig are provided.
    if (groupId && groupConfig) {
        legendConfig.onClick = createGroupLegendClickHandler(groupId, groupConfig);
    }

    return {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: { left: 0, right: 0, top: 0, bottom: 0 }
        },
        interaction: {
            intersect: false,
            mode: "index",
        },
        scales: {
            x: {
                title: { display: false },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45,
                    autoSkipPadding: 10,
                },
                min: Math.max(0, chartCommits.length - CONFIG.defaultVisibleCommits),
            },
            y: {
                title: {
                    display: true,
                    text: CONFIG.yAxisLabel,
                },
                beginAtZero: true,
            },
        },
        plugins: {
            verticalLine: {},
            legend: legendConfig,
            tooltip: createTooltipConfig(chartCommits, tooltipElementId),
        },
        onClick: (event, elements) => handleChartClick(elements, chartCommits),
    };
}

/**
 * Handles click events on chart data points.
 */
function handleChartClick(elements, chartCommits) {
    if (elements.length > 0) {
        const index = elements[0].index;
        const commit = chartCommits[index];
        if (commit?.url) {
            window.open(commit.url, "_blank");
        }
    }
}

/**
 * Creates a Chart.js instance.
 */
function createChartInstance(canvasId, chartCommits, seriesData, groupId, groupConfig, tooltipElementId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext("2d");
    const datasets = createDatasets(seriesData, groupConfig);

    return new Chart(ctx, {
        type: "line",
        data: {
            labels: chartCommits.map(c => c.id.slice(0, 7)),
            datasets: datasets,
        },
        options: createChartOptions(chartCommits, tooltipElementId, groupId, groupConfig),
    });
}

// ============================================================================
// Timeline Control Functions
// ============================================================================

/**
 * Creates chart context for timeline state management.
 */
function createChartContext(totalCommits) {
    return {
        totalCommits,
        minWindowSize: CONFIG.minWindowSize,
        maxWindowSize: totalCommits,
        defaultWindowSize: CONFIG.defaultVisibleCommits,
        currentWindowSize: Math.min(CONFIG.defaultVisibleCommits, totalCommits),
        currentPosition: totalCommits
    };
}

/**
 * Updates scrollbar dimensions to match current window size.
 */
function updateScrollbarDimensions(elements, chartContext) {
    const containerWidth = elements.scrollbarContainer.clientWidth;
    const ratio = chartContext.totalCommits / chartContext.currentWindowSize;
    const contentWidth = Math.max(containerWidth * ratio, containerWidth * 1.01);
    elements.scrollbarContent.style.width = `${contentWidth}px`;
}

/**
 * Updates chart view and UI to reflect current state.
 */
function updateChartView(elements, chartContext, chartInstance, updateScrollbar) {
    const windowSize = chartContext.currentWindowSize;
    const position = chartContext.currentPosition;

    const endIndex = Math.min(position, chartContext.totalCommits);
    const startIndex = Math.max(0, endIndex - windowSize);

    chartInstance.options.scales.x.min = startIndex;
    chartInstance.options.scales.x.max = endIndex - 1;
    chartInstance.update('none');

    elements.controlInfoText.textContent =
        `Showing commits ${startIndex + 1}-${endIndex} of ${chartContext.totalCommits} (${windowSize} visible)`;

    if (updateScrollbar) {
        updateScrollbarDimensions(elements, chartContext);
        const scrollPercentage = (endIndex - windowSize) / (chartContext.totalCommits - windowSize);
        elements.scrollbarContainer.scrollLeft = scrollPercentage *
            (elements.scrollbarContent.clientWidth - elements.scrollbarContainer.clientWidth);
    }
}

/**
 * Sets up scrollbar event handler.
 */
function setupScrollbarHandler(elements, chartContext, chartInstance) {
    elements.scrollbarContainer.addEventListener("scroll", () => {
        const scrollLeft = elements.scrollbarContainer.scrollLeft;
        const maxScroll = elements.scrollbarContent.clientWidth - elements.scrollbarContainer.clientWidth;
        const scrollPercentage = maxScroll > 0 ? scrollLeft / maxScroll : 0;

        const windowSize = chartContext.currentWindowSize;
        const newPosition = Math.round(windowSize + scrollPercentage * (chartContext.totalCommits - windowSize));
        chartContext.currentPosition = Math.min(chartContext.totalCommits, Math.max(windowSize, newPosition));

        updateChartView(elements, chartContext, chartInstance, false);
    });
}

/**
 * Sets up zoom button click handlers.
 */
function setupZoomButtons(elements, chartContext, chartInstance) {
    const zoom = (step, direction) => {
        const currentWindowSize = chartContext.currentWindowSize;

        let newWindowSize;
        if (direction > 0) {
            newWindowSize = Math.ceil((currentWindowSize + 1) / step) * step;
        } else {
            newWindowSize = Math.floor((currentWindowSize - 1) / step) * step;
        }
        newWindowSize = Math.max(chartContext.minWindowSize, Math.min(chartContext.maxWindowSize, newWindowSize));

        const currentStart = chartContext.currentPosition - currentWindowSize;
        const currentCenter = currentStart + currentWindowSize / 2;
        chartContext.currentWindowSize = newWindowSize;
        chartContext.currentPosition = Math.min(
            chartContext.totalCommits,
            Math.max(newWindowSize, Math.round(currentCenter + newWindowSize / 2))
        );

        updateChartView(elements, chartContext, chartInstance, true);
    };

    elements.zoomInSmallBtn?.addEventListener("click", () => zoom(25, -1));
    elements.zoomInLargeBtn?.addEventListener("click", () => zoom(250, -1));
    elements.zoomOutSmallBtn?.addEventListener("click", () => zoom(25, 1));
    elements.zoomOutLargeBtn?.addEventListener("click", () => zoom(250, 1));
}

/**
 * Sets up mouse wheel pan handler.
 */
function setupWheelPanHandler(elements, chartContext, chartInstance) {
    elements.chartCanvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        const delta = Math.sign(e.deltaY);
        const panAmount = Math.max(1, Math.round(chartContext.currentWindowSize * 0.1));

        chartContext.currentPosition = Math.min(
            chartContext.totalCommits,
            Math.max(chartContext.currentWindowSize, chartContext.currentPosition + delta * panAmount)
        );

        updateChartView(elements, chartContext, chartInstance, true);
    });
}

/**
 * Initializes timeline controls for a chart.
 */
function initializeTimelineControls(chartInstance, chartCommits, prefix) {
    const elements = {
        scrollbarContainer: document.getElementById(`${prefix}scrollbar-container`),
        scrollbarContent: document.getElementById(`${prefix}scrollbar-content`),
        controlInfoText: document.getElementById(`${prefix}info`),
        chartCanvas: document.getElementById(`${prefix}canvas`),
        zoomInSmallBtn: document.getElementById(`${prefix}zoom-in-small`),
        zoomInLargeBtn: document.getElementById(`${prefix}zoom-in-large`),
        zoomOutSmallBtn: document.getElementById(`${prefix}zoom-out-small`),
        zoomOutLargeBtn: document.getElementById(`${prefix}zoom-out-large`),
    };

    const chartContext = createChartContext(chartCommits.length);

    if (chartContext.totalCommits === 0) {
        if (elements.scrollbarContainer) elements.scrollbarContainer.style.display = "none";
        if (elements.controlInfoText) elements.controlInfoText.textContent = "No data available";
        return;
    }

    if (chartContext.totalCommits === 1) {
        if (elements.scrollbarContainer) elements.scrollbarContainer.style.display = "none";
        chartContext.maxWindowSize = 1;
    }

    setupScrollbarHandler(elements, chartContext, chartInstance);
    setupZoomButtons(elements, chartContext, chartInstance);
    setupWheelPanHandler(elements, chartContext, chartInstance);

    updateChartView(elements, chartContext, chartInstance, true);
}

// ============================================================================
// Chart Instance Storage
// ============================================================================

/**
 * Global storage for chart instances. Key is the chart prefix (groupId-chartName-).
 */
const chartInstances = new Map();

/**
 * Tracks hidden series per group. When a legend item is clicked, all charts in the group update.
 * Structure: groupId -> Set<seriesName>
 */
const hiddenSeries = new Map();

/**
 * Creates a legend click handler that syncs visibility across all charts in a group.
 */
function createGroupLegendClickHandler(groupId, groupConfig) {
    return function(e, legendItem, legend) {
        const seriesName = legendItem.text;
        const hidden = hiddenSeries.get(groupId) || new Set();

        // Toggle visibility.
        if (hidden.has(seriesName)) {
            hidden.delete(seriesName);
        } else {
            hidden.add(seriesName);
        }
        hiddenSeries.set(groupId, hidden);

        // Update all charts in this group (no animation for performance with many charts).
        for (const [prefix, chart] of chartInstances.entries()) {
            if (prefix.startsWith(groupId + '-')) {
                const datasetIndex = groupConfig.seriesNames.indexOf(seriesName);
                if (datasetIndex >= 0) {
                    chart.setDatasetVisibility(datasetIndex, !hidden.has(seriesName));
                    chart.update('none');
                }
            }
        }
    };
}

// ============================================================================
// Main Rendering Functions
// ============================================================================

/**
 * Creates an empty Chart.js instance (structure only, no data).
 * This allows the chart to be visible immediately while data loads.
 */
function createEmptyChart(groupId, chartName, groupConfig) {
    const prefix = makeId(groupId, chartName, '');
    const canvas = document.getElementById(`${prefix}canvas`);
    if (!canvas) return null;

    const ctx = canvas.getContext("2d");
    const datasets = createEmptyDatasets(groupConfig);

    const chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: true },
                y: {
                    display: true,
                    title: { display: true, text: CONFIG.yAxisLabel },
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: { position: "top" },
            },
        },
    });

    chartInstances.set(prefix, chart);
    return chart;
}

/**
 * Updates an existing chart with data.
 */
async function renderChart(wasm, groupId, chartName, groupConfig, commits) {
    const prefix = makeId(groupId, chartName, '');

    try {
        const chartData = await loadChartData(wasm, groupId, chartName);
        const { seriesData, chartCommits } = processChartData(chartData, commits, groupConfig);

        // Store chart data for group summary calculation.
        storeChartData(groupId, chartName, seriesData);

        // Get the existing chart instance.
        let chartInstance = chartInstances.get(prefix);

        if (chartInstance) {
            // Update existing chart with new data.
            chartInstance.data.labels = chartCommits.map(c => c.id.slice(0, 7));
            chartInstance.data.datasets = createDatasets(seriesData, groupConfig);

            // Update options for tooltips, click handling, and linked legend.
            chartInstance.options = createChartOptions(chartCommits, `${prefix}tooltip`, groupId, groupConfig);

            chartInstance.update('none');
        } else {
            // Fallback: create new chart if empty one wasn't created.
            chartInstance = createChartInstance(
                `${prefix}canvas`,
                chartCommits,
                seriesData,
                groupId,
                groupConfig,
                `${prefix}tooltip`
            );
            if (chartInstance) {
                chartInstances.set(prefix, chartInstance);
            }
        }

        if (chartInstance) {
            initializeTimelineControls(chartInstance, chartCommits, prefix);
        }

        return true;
    } catch (error) {
        console.error(`Failed to render chart ${groupId}/${chartName}:`, error);
        const infoEl = document.getElementById(`${prefix}info`);
        if (infoEl) infoEl.textContent = `Error: ${error.message}`;
        return false;
    }
}

/**
 * Creates chart placeholders for a group (immediate, no data loading).
 */
function createChartPlaceholders(groupId, groupConfig) {
    const chartsContainer = document.getElementById(`${groupId}-charts`);
    if (!chartsContainer) return;

    chartsContainer.innerHTML = groupConfig.charts
        .map(chartName => createChartHTML(groupId, chartName))
        .join('');
}

/**
 * Creates empty Chart.js instances for all charts in a group.
 * This makes charts visible immediately (with legend but no data).
 */
function createEmptyCharts(groupId, groupConfig) {
    for (const chartName of groupConfig.charts) {
        createEmptyChart(groupId, chartName, groupConfig);
    }
}

/**
 * Waits one frame (16ms) for the browser to paint.
 *
 * This is used to yield control back to the browser between chart renders, so each chart appears
 * progressively rather than all at once.
 */
function waitForPaint() {
    return new Promise(resolve => setTimeout(resolve, 16));
}

// ============================================================================
// Main Orchestration
// ============================================================================

/**
 * Main function that orchestrates the entire benchmark page initialization.
 */
async function main() {
    try {
        // Register the vertical line plugin once.
        Chart.register(createVerticalLinePlugin());

        // Step 1: Create all group containers immediately from JS config.
        // This happens BEFORE loading any data so users see the UI instantly.
        const container = document.getElementById("benchmarks-container");
        if (!container) {
            throw new Error("Benchmarks container not found");
        }

        let groupsHTML = '';
        const allGroups = Object.entries(BENCHMARK_GROUPS);
        for (const [groupId, groupConfig] of allGroups) {
            groupsHTML += createGroupHTML(groupId, groupConfig);
        }
        container.innerHTML = groupsHTML;

        // Step 2: Create all chart placeholders (HTML structure).
        for (const [groupId, groupConfig] of allGroups) {
            createChartPlaceholders(groupId, groupConfig);
        }

        // Step 3: Create empty Chart.js instances (visible charts with no data).
        for (const [groupId, groupConfig] of allGroups) {
            createEmptyCharts(groupId, groupConfig);
        }

        // Step 4: Set up collapsible behavior.
        setupCollapsibleBenchmarks();

        // Step 5: Show status - UI is ready, data loading in background.
        const totalCharts = allGroups.reduce((n, [_, c]) => n + c.charts.length, 0);
        setStatus(`Loading ${totalCharts} charts...`, "loading");

        // Step 6: Yield to browser to paint the UI before loading data.
        await waitForPaint();

        // Step 7: NOW load WASM and data (charts already visible).
        const wasm = await loadWasmModule();
        const summary = await loadBenchmarkSummary(wasm);

        console.log("Available groups from server:", Object.keys(summary.groups));
        console.log("Configured groups:", Object.keys(BENCHMARK_GROUPS));

        // Step 8: Filter to groups that exist on server.
        const activeGroups = allGroups.filter(([groupId]) => {
            const exists = !!summary.groups[groupId];
            if (!exists) {
                console.warn(`Group '${groupId}' not found in server data, skipping`);
            }
            return exists;
        });

        // Step 9: Update charts with data one by one (strictly sequential).
        // Charts are already visible (empty), now we fill in the data.
        let loadedCharts = 0;
        for (const [groupId, groupConfig] of activeGroups) {
            const statusEl = document.getElementById(`${groupId}-status`);
            const groupTotal = groupConfig.charts.length;
            let groupLoaded = 0;

            for (const chartName of groupConfig.charts) {
                const success = await renderChart(wasm, groupId, chartName, groupConfig, summary.commits);
                if (success) {
                    groupLoaded++;
                    loadedCharts++;
                }

                // Update group status.
                if (statusEl) {
                    statusEl.textContent = `${groupLoaded}/${groupTotal} charts`;
                }

                // Wait for browser to paint before rendering next chart.
                await waitForPaint();
            }

            // Render group summary after all charts are loaded.
            renderGroupSummary(groupId, groupConfig);

            console.log(`Rendered ${groupLoaded}/${groupTotal} charts in ${groupId}`);
        }

        setStatus(`Loaded ${loadedCharts} charts across ${activeGroups.length} groups`, "success");

    } catch (error) {
        console.error("Error:", error);
        setStatus(`Error: ${error.message}`, "error");
    }
}

main();
