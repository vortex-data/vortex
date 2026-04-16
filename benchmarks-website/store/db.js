function normalizeValue(value) {
  if (typeof value === "bigint") {
    const asNumber = Number(value);
    return Number.isSafeInteger(asNumber) ? asNumber : value.toString();
  }

  if (Array.isArray(value)) {
    return value.map(normalizeValue);
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, normalizeValue(child)]),
    );
  }

  return value;
}

export async function queryRows(connection, sql, values) {
  const result = await connection.run(sql, values);
  const rows = await result.getRowObjectsJS();
  return rows.map((row) => normalizeValue(row));
}

export async function withConnection(instance, fn) {
  const connection = await instance.connect();
  try {
    return await fn(connection);
  } finally {
    connection.closeSync();
  }
}
