import { ensureInitialized, getStore } from "../../lib/server/store.js";

export async function GET() {
  await ensureInitialized();
  const store = getStore();

  if (!store.metadata) {
    return new Response(JSON.stringify({ error: "Loading" }), {
      status: 503,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  }

  return new Response(JSON.stringify(store.metadata), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
}
