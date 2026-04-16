export function firstLine(message) {
  return String(message || "").split("\n")[0] || "";
}
