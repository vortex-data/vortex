#!/usr/bin/env node

import fs from "fs/promises";
import path from "path";
import { buildBootstrapSql } from "../store/sql.js";

function parseArgs(argv) {
  const options = {
    dataPath: null,
    commitsPath: null,
    outputPath: null,
    placeholders: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--data-path":
        options.dataPath = argv[++i] || null;
        break;
      case "--commits-path":
        options.commitsPath = argv[++i] || null;
        break;
      case "--output":
        options.outputPath = argv[++i] || null;
        break;
      case "--placeholders":
        options.placeholders = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function printHelp() {
  process.stdout.write(`Usage:
  node scripts/export-bootstrap-sql.js --data-path <path> --commits-path <path> [--output <path>]
  node scripts/export-bootstrap-sql.js --placeholders [--output <path>]

Options:
  --data-path <path>      Path to data.json.gz
  --commits-path <path>   Path to commits.json
  --output <path>         Write SQL to a file instead of stdout
  --placeholders          Emit __DATA_PATH__ / __COMMITS_PATH__ placeholders
  --help, -h              Show this help
`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  const dataPath = options.placeholders ? "__DATA_PATH__" : options.dataPath;
  const commitsPath = options.placeholders
    ? "__COMMITS_PATH__"
    : options.commitsPath;

  if (!dataPath || !commitsPath) {
    printHelp();
    process.stderr.write(
      "\nerror: either provide --data-path and --commits-path, or use --placeholders\n",
    );
    process.exit(1);
  }

  const sql = `${buildBootstrapSql(dataPath, commitsPath)}\n`;

  if (options.outputPath) {
    const outputPath = path.resolve(options.outputPath);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, sql, "utf8");
    process.stdout.write(`${outputPath}\n`);
    return;
  }

  process.stdout.write(sql);
}

main().catch((error) => {
  process.stderr.write(`${error.message}\n`);
  process.exit(1);
});
