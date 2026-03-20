import { ensureInitialized, handleDataRequest } from "../../../lib/server/store.js";

export async function GET({ params, request }) {
  await ensureInitialized();

  const pathParts = params.path.split("/");
  const group = decodeURIComponent(pathParts[0] || "");
  const chart = decodeURIComponent(pathParts.slice(1).join("/") || "");

  const url = new URL(request.url);
  const queryParams = {
    start: url.searchParams.get("start"),
    end: url.searchParams.get("end"),
    last: url.searchParams.get("last"),
    startIdx: url.searchParams.has("startIdx")
      ? url.searchParams.get("startIdx")
      : undefined,
    endIdx: url.searchParams.has("endIdx")
      ? url.searchParams.get("endIdx")
      : undefined,
  };

  const result = handleDataRequest(group, chart, queryParams);

  const headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  };

  if (result.error) {
    return new Response(JSON.stringify({ error: result.error }), {
      status: result.status,
      headers,
    });
  }

  return new Response(JSON.stringify(result.data), {
    status: 200,
    headers,
  });
}
