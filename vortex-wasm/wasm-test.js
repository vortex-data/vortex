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
 * Loads and initializes the WASM module.
 */
async function loadWasmModule() {
    setStatus("Loading WASM module...");
    const wasm = await import(CONFIG.wasmModulePath);
    await wasm.default();
    console.log("WASM loaded:", wasm.get_version());
    return wasm;
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
                    <span>${groupConfig.charts.length} charts</span>
                </div>
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
            <div class="summary-section">
                <div id="${prefix}summary" class="scores-list"></div>
                <p class="scores-explanation">
                    Query time | Ratio to fastest (lower is better)
                </p>
            </div>
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
 * Renders the summary table showing latest benchmark results.
 */
function renderSummary(summaryElementId, summary, groupConfig) {
    const summaryList = document.getElementById(summaryElementId);
    if (!summaryList) return;

    summaryList.innerHTML = "";

    if (summary.results.length === 0) {
        summaryList.innerHTML = '<div class="score-item">No data available</div>';
        return;
    }

    summary.results.forEach(([name, time], index) => {
        const ratio = time / summary.fastestTime;
        const item = document.createElement("div");
        item.className = "score-item";
        item.innerHTML = `
            <span class="score-rank">#${index + 1}</span>
            <span class="score-series" style="color: ${groupConfig.seriesColors[name]}">${name}</span>
            <div class="score-metrics">
                <span class="score-runtime">${formatTime(time)}</span>
                <span class="score-ratio">${ratio.toFixed(2)}x</span>
            </div>
        `;
        summaryList.appendChild(item);
    });
}

/**
 * Sets up collapsible benchmark sections.
 */
function setupCollapsibleBenchmarks() {
    document.querySelectorAll('.benchmark-header').forEach(header => {
        header.addEventListener('click', () => {
            const benchmarkSet = header.closest('.benchmark-set');
            benchmarkSet.classList.toggle('collapsed');
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
function createChartOptions(chartCommits, tooltipElementId) {
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
            legend: { position: "top" },
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
function createChartInstance(canvasId, chartCommits, seriesData, groupConfig, tooltipElementId) {
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
        options: createChartOptions(chartCommits, tooltipElementId),
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
// Main Rendering Functions
// ============================================================================

/**
 * Renders a single chart within a group.
 */
async function renderChart(wasm, groupId, chartName, groupConfig, commits) {
    const prefix = makeId(groupId, chartName, '');

    try {
        const chartData = await loadChartData(wasm, groupId, chartName);
        const { seriesData, chartCommits } = processChartData(chartData, commits, groupConfig);
        const summaryStats = calculateSummary(seriesData);

        renderSummary(`${prefix}summary`, summaryStats, groupConfig);

        const chartInstance = createChartInstance(
            `${prefix}canvas`,
            chartCommits,
            seriesData,
            groupConfig,
            `${prefix}tooltip`
        );

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
 * Loads all charts in a group in the background.
 * Returns a promise that resolves when all charts are loaded.
 */
async function loadGroupCharts(wasm, groupId, groupConfig, commits) {
    let successCount = 0;

    // Load charts sequentially to avoid overwhelming the server.
    for (const chartName of groupConfig.charts) {
        const success = await renderChart(wasm, groupId, chartName, groupConfig, commits);
        if (success) successCount++;

        // Yield to browser to paint after each chart.
        await new Promise(resolve => setTimeout(resolve, 0));
    }

    console.log(`Rendered ${successCount}/${groupConfig.charts.length} charts in ${groupId}`);
    return successCount;
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

        // Load WASM module.
        const wasm = await loadWasmModule();

        // Load summary (fast - metadata only).
        const summary = await loadBenchmarkSummary(wasm);

        console.log("Available groups from server:", Object.keys(summary.groups));
        console.log("Configured groups:", Object.keys(BENCHMARK_GROUPS));

        // Get the container for all groups.
        const container = document.getElementById("benchmarks-container");
        if (!container) {
            throw new Error("Benchmarks container not found");
        }
        console.log("Found container:", container);

        // Step 1: Create all group containers immediately.
        let groupsHTML = '';
        const activeGroups = [];
        for (const [groupId, groupConfig] of Object.entries(BENCHMARK_GROUPS)) {
            console.log(`Checking group '${groupId}':`, summary.groups[groupId] ? "found" : "NOT FOUND");
            if (summary.groups[groupId]) {
                groupsHTML += createGroupHTML(groupId, groupConfig);
                activeGroups.push([groupId, groupConfig]);
            } else {
                console.warn(`Group '${groupId}' not found in server data, skipping`);
            }
        }
        console.log("Active groups:", activeGroups.map(g => g[0]));
        console.log("Groups HTML length:", groupsHTML.length);
        container.innerHTML = groupsHTML;

        // Step 2: Create all chart placeholders immediately.
        for (const [groupId, groupConfig] of activeGroups) {
            createChartPlaceholders(groupId, groupConfig);
        }

        // Step 3: Set up collapsible behavior.
        setupCollapsibleBenchmarks();

        // Step 4: Show status - UI is ready, data loading in background.
        setStatus(`Loading ${activeGroups.reduce((n, [_, c]) => n + c.charts.length, 0)} charts...`, "loading");

        // Step 5: Yield to browser to paint the UI before loading data.
        await new Promise(resolve => setTimeout(resolve, 0));

        // Step 6: Load chart data in background.
        // Load groups in parallel, charts within each group sequentially.
        const loadPromises = activeGroups.map(([groupId, groupConfig]) =>
            loadGroupCharts(wasm, groupId, groupConfig, summary.commits)
        );

        // Wait for all to complete and update status.
        const results = await Promise.all(loadPromises);
        const totalCharts = results.reduce((a, b) => a + b, 0);
        setStatus(`Loaded ${totalCharts} charts across ${activeGroups.length} groups`, "success");

    } catch (error) {
        console.error("Error:", error);
        setStatus(`Error: ${error.message}`, "error");
    }
}

main();
