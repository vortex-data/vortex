'use strict';
window.initAndRender = (function () {
    function stringToColor(str) {
        // Random colours are generally pretty disgusting...
        const MAP = {
            "arrow": '#58067e',
            "parquet": '#ef7f1d',
            "vortex-file-compressed": '#23d100',
        };

        if (MAP[str]) {
            return MAP[str];
        }

        var hash = new Hashes.MD5().hex(str)

        // Return a CSS color string
        const hexColor = hash.slice(0, 2) + hash.slice(14, 16) + hash.slice(30, 32);
        return `#${hexColor}`;
    }

    function downloadAndGroupData(data, commit_metadata) {
        // It's desirable for all our graphs to line up in terms of X-axis.
        // As such, we collect all unique {commit,entry} first, and then assign
        // data points to them for each graph. Commits are sorted by date.
        const commits = [];
        Object.values(commit_metadata).sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp)).forEach((commit, commitSortedIndex) => {
            commit.sortedIndex = commitSortedIndex;
            commits.push(commit);
        });

        // Prepare data points for charts
        let groups = {
            "Random Access": new Map(),
            "Compression": new Map(),
            "TPC-H (NVME)": new Map(),
            "TPC-H (S3)": new Map(),
            "Clickbench": new Map(),
        };

        let uncategorizable_names = new Set();
        let missing_commits = new Set();

        for (let benchmark_result of data) {
            let commit_id = benchmark_result.commit_id;
            benchmark_result["commit"] = commit_metadata[commit_id];
            if (!benchmark_result["commit"]) {
                missing_commits.add(commit_id)
                benchmark_result["commit"] = commit_metadata[commit_id] = {
                    "author":{"email":"daniel.zidan.king@gmail.com","name":"Dan King"},
                    "committer":{"email":"noreply@github.com","name":"GitHub"},
                    "id":commit_id,
                    "message":"!! This commit is missing from commits.json !!",
                    "timestamp":"1970-01-01T00:00:00Z",
                    "tree_id":null,
                    "url":"https://github.com/spiraldb/vortex/commit/" + commit_id
                }
            }

            let {name, unit, value, commit} = benchmark_result;
            let storage = benchmark_result.storage;
            let group = undefined;

            if (name.startsWith("random-access/")) {
                group = groups["Random Access"];
            } else if (name.includes("compress time/")) {
                group = groups["Compression"];
            } else if (name.startsWith("tpch_q")) {
                if (storage === undefined || storage == "nvme") {
                    group = groups["TPC-H (NVME)"];
                } else {
                    group = groups["TPC-H (S3)"];
                }
            } else if (name.startsWith("clickbench")) {
                group = groups["Clickbench"];
            } else {
                uncategorizable_names.add(name)
                continue
            }


            // Normalize name and units
            let [q, seriesName] = name.split("/");
            if (seriesName.endsWith(" throughput")) {
                seriesName = seriesName.slice(0, seriesName.length - " throughput".length);
                q = q.replace("time", "throughput");
            } else if (seriesName.endsWith("throughput")) {
                seriesName = seriesName.slice(0, seriesName.length - "throughput".length);
                q = q.replace("time", "throughput");
            }

            let prettyQ = q.replace("_", " ")
                .toUpperCase()
                .replace("VORTEX:RAW SIZE", "VORTEX COMPRESSION RATIO");
            if (prettyQ.includes("PARQUET-UNC")) {
                return
            }

            const is_nanos = unit === "ns/iter" || unit === "ns";
            const is_bytes = unit === "bytes";
            const is_throughput = unit === "bytes/ns";

            let sort_position = (q.slice(0, 4) == "tpch") ? parseInt(prettyQ.split(" ")[1].substring(1), 10) : 0;

            let arr = group.get(prettyQ);
            if (arr === undefined) {
                group.set(prettyQ, {
                    sort_position,
                    commits,
                    unit: is_nanos ? "ms/iter" : (is_bytes ? "MiB" : (is_throughput ? "MiB/s" : unit)),
                    series: new Map(),
                });
                arr = group.get(prettyQ);
            }

            let series = arr.series.get(seriesName);
            if (series === undefined) {
                arr.series.set(seriesName, new Array(commits.length).fill(null));
                series = arr.series.get(seriesName);
            }

            series[commit.sortedIndex] = {
                range: "this was the range",
                value: is_nanos ? value / 1_000_000 : (is_bytes ? value / 1_048_576 : (is_throughput ? value * 1_000_000_000 / 1_048_576 : value))
            };
        }

        function sortByPositionThenName(a, b) {
            let position_compare = a[1].sort_position - b[1].sort_position
            if (position_compare !== 0) {
                return position_compare
            }
            return a[0].localeCompare(b[0]);
        }

        Object.entries(groups).forEach(pair => {
            let [name, charts] = pair;
            groups[name] = new Map([...charts.entries()].sort(sortByPositionThenName));
        });

        console.warn(
            "these commits were missing from commits.json so the commit message is missing and the datetime is set to 1970-01-01T00:00:00Z",
            missing_commits
        );
        console.warn(
            "could not categorizes benchmarks with these names, they will not be shown: ",
            uncategorizable_names
        );

        return Object.keys(groups).map(name => ({
            name,
            dataSet: groups[name],
        }));
    }

    function renderAllCharts(dataSets, keptGroups) {
        var charts = [];

        function renderChart(parent, name, dataset, hiddenDatasets, removedDatasets, renamedDatasets) {
            const canvasContainer = document.createElement('div');
            parent.appendChild(canvasContainer);

            const canvas = document.createElement('canvas');
            canvas.className = 'benchmark-chart';
            canvasContainer.appendChild(canvas);

            const data = {
                labels: dataset.commits.map(commit => commit.id.slice(0, 7)),
                datasets: Array.from(dataset.series).filter(([name, benches]) => {
                    return removedDatasets === undefined || !removedDatasets.has(name)
                }).map(([name, benches]) => {
                    const color = stringToColor(name);
                    const renamedName = (renamedDatasets == undefined) ? name : (renamedDatasets[name] || name);
                    return {
                        label: renamedName,
                        data: benches.map(b => b ? b.value : null),
                        borderColor: color,
                        backgroundColor: color + '60', // Add alpha for #rrggbbaa
                        hidden: hiddenDatasets !== undefined && hiddenDatasets.has(name),
                    };
                }),
            };
            const y_axis_scale = {
                title: {
                    display: true,
                    text: dataset.commits.length > 0 ? dataset.unit : '',
                },
                suggestedMin: 0,
            }

            if (name.includes("COMPRESS") && name.includes("THROUGHPUT") && dataset.unit == "MiB/s") {
                y_axis_scale.suggestedMax = 1024;
                y_axis_scale.max = 1024;
            }

            if (name.includes("DECOMPRESS") && name.includes("THROUGHPUT") && dataset.unit == "MiB/s") {
                y_axis_scale.suggestedMax = 4096;
                y_axis_scale.max = 4096;
            }

            const options = {
                responsive: true,
                maintainAspectRatio: false,
                spanGaps: true,
                pointStyle: 'crossRot',
                elements: {
                    line: {
                        borderWidth: 1,
                    },

                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: name,
                            padding: {bottom: 50},
                        },
                        // By default, show the last 50 commits
                        min: Math.max(0, dataset.commits.length - 50),
                    },
                    y: y_axis_scale,
                },
                plugins: {
                    zoom: {
                        zoom: {
                            wheel: {enabled: true},
                            mode: 'x',
                            drag: {enabled: true}
                        }
                    },
                    legend: {
                        display: true,
                        onClick: function (e, legendItem) {
                            var index = legendItem.datasetIndex;

                            var wasVisible = this.chart.isDatasetVisible(index);
                            var datasetLabel = this.chart.data.datasets[index].label;
                            var clickedChart = this.chart;

                            charts.forEach(function (chart) {
                                chart.data.datasets.forEach(function (ds, idx) {
                                    if (ds.label === datasetLabel) {
                                        chart.getDatasetMeta(idx).hidden = wasVisible;
                                    }
                                });

                                chart.update();
                            });
                        }
                    },
                    tooltip: {
                        callbacks: {
                            footer: items => {
                                const {dataIndex} = items[0];
                                const commit = dataset.commits[dataIndex];
                                return commit.message.split("\n")[0] + "\n" + commit.author.name + " <" + commit.author.email + ">";
                            }
                        }
                    }
                },
                onClick: (_mouseEvent, activeElems) => {
                    if (activeElems.length === 0) {
                        return;
                    }
                    // XXX: Undocumented. How can we know the index?
                    const index = activeElems[0].index;
                    const url = dataset.commits[index].url;
                    window.open(url, '_blank');
                },
            };

            return new Chart(canvas, {
                type: 'line',
                data,
                options,
            });
        }

        function renderBenchSet(name, benchSet, main, toc, groupFilterSettings) {
            const {keptCharts, hiddenDatasets, removedDatasets, renamedDatasets} = (
                groupFilterSettings === undefined
                    ? {keptCharts: undefined, hiddenDatasets: undefined, removedDatasets: undefined, renamedDatasets: undefined}
                    : groupFilterSettings);
            const setElem = document.createElement('div');
            setElem.className = 'benchmark-set';
            main.appendChild(setElem);

            const h1id = name.replace(" ", "_");
            const nameElem = document.createElement('h1');
            nameElem.id = h1id;
            nameElem.className = 'benchmark-title';
            nameElem.textContent = name;
            setElem.appendChild(nameElem);

            const tocLi = document.createElement('li');
            const tocLink = document.createElement('a');
            tocLink.href = '#' + h1id;
            tocLink.innerHTML = name;
            tocLi.appendChild(tocLink);
            toc.appendChild(tocLi);

            const graphsElem = document.createElement('div');
            graphsElem.className = 'benchmark-graphs';
            setElem.appendChild(graphsElem);

            if (keptCharts == undefined) {
                for (const [benchName, benches] of benchSet.entries()) {
                    charts.push(renderChart(graphsElem, benchName, benches, hiddenDatasets, removedDatasets, renamedDatasets))
                }
            } else {
                for (const benchName of keptCharts) {
                    const benches = benchSet.get(benchName)
                    if (benches) {
                        charts.push(renderChart(graphsElem, benchName, benches, hiddenDatasets, removedDatasets, renamedDatasets))
                    }
                }
            }
        }

        const main = document.getElementById('main');
        const toc = document.getElementById('toc');
        if (keptGroups === undefined) {
            for (const {name, dataSet} of dataSets) {
                renderBenchSet(name, dataSet, main, toc, undefined);
            }
        } else {
            const dataSetsMap = new Map(dataSets.map(({name, dataSet}) => [name, dataSet]));
            for (const [name, groupFilterSettings] of keptGroups) {
                const dataSet = dataSetsMap.get(name);
                renderBenchSet(name, dataSet, main, toc, groupFilterSettings);
            }
        }
    }

    function parse_jsonl(jsonl) {
        return jsonl
            .split('\n')
            .filter(line => line.trim().length != 0)
            .map(line => JSON.parse(line))
    }

    function initAndRender(keptGroups) {
        // let data = fetch('https://vortex-benchmark-results-database.s3.amazonaws.com/data.json')
        let data = fetch('data.json')
            .then(response => response.text())
            .then(parse_jsonl)
            .catch(error => console.error('unable to load data.json:', error));
        let commit_metadata = fetch('https://vortex-benchmark-results-database.s3.amazonaws.com/commits.json')
            .then(response => response.text())
            .then(parse_jsonl)
            .then(commit_metadatas => {
                return commit_metadatas.reduce((dict, commit_metadata) => {
                    dict[commit_metadata.id] = commit_metadata;
                    return dict;
                }, {})
            })
            .catch(error => console.error('unable to load commits.json:', error));
        Promise.all([data, commit_metadata]).then(pair => renderAllCharts(downloadAndGroupData(pair[0], pair[1]), keptGroups))
    };

    return initAndRender;
})();
