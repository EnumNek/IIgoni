// Работа с графом на клиенте: применение изменений (для мгновенного отображения до пересчёта),
// подписи изменений и форматирование величин.
import type { ChangeIn, Network, NetEdge } from "./types";

/** Те же правила, что backend/app/services/changes.py: изменения применяются по порядку. */
export function applyChanges(net: Network, changes: ChangeIn[]): Network {
  const edges = net.edges.map((e) => ({ ...e }));
  const nodes = net.nodes.map((n) => ({ ...n }));
  const eById = new Map(edges.map((e) => [e.id, e]));
  const nById = new Map(nodes.map((n) => [n.id, n]));
  const original = new Map(net.nodes.map((n) => [n.id, n.type]));
  for (const ch of changes) {
    const e = eById.get(ch.target_id);
    if (ch.type === "lanes" && e) e.lanes = Number(ch.value);
    else if (ch.type === "speed" && e) e.speed = Number(ch.value);
    else if (ch.type === "close" && e) e.closed = true;
    else if (ch.type === "open" && e) e.closed = false;
    else if (ch.type === "traffic_light") {
      const n = nById.get(ch.target_id);
      if (!n) continue;
      if (ch.value === false) n.type = original.get(n.id) === "traffic_light" ? "priority" : original.get(n.id)!;
      else n.type = "traffic_light";
    }
  }
  return { ...net, edges, nodes };
}

/** Встречное ребро той же улицы (from/to наоборот). */
export function reverseEdge(net: Network, edge: NetEdge): NetEdge | undefined {
  return net.edges.find((e) => e.from === edge.to && e.to === edge.from && e.id !== edge.id);
}

export function edgeTitle(e: NetEdge | undefined): string {
  if (!e) return "участок";
  return e.name || `участок ${e.id}`;
}

export const kmh = (ms: number | null | undefined) => (ms == null ? "—" : `${(ms * 3.6).toFixed(1)} км/ч`);
export const pct = (x: number | null | undefined, digits = 0) => (x == null ? "—" : `${(x * 100).toFixed(digits)} %`);

export function duration(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return "—";
  if (sec < 60) return `${sec.toFixed(1)} с`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m} мин ${s.toString().padStart(2, "0")} с`;
}

export function describeChange(ch: ChangeIn, net: Network | null): string {
  const e = net?.edges.find((x) => x.id === ch.target_id);
  const title = edgeTitle(e);
  switch (ch.type) {
    case "lanes":
      return `${title}: полос → ${ch.value}`;
    case "speed":
      return `${title}: скорость → ${Math.round(Number(ch.value) * 3.6)} км/ч`;
    case "close":
      return `${title}: закрыт`;
    case "open":
      return `${title}: открыт`;
    case "traffic_light": {
      // Перекрёсток называем по улицам, которые в нём сходятся
      const names = [...new Set((net?.edges ?? []).filter((x) => x.to === ch.target_id && x.name).map((x) => x.name))];
      const where = names.length ? `Перекрёсток ${names.slice(0, 2).join(" / ")}` : `Узел ${ch.target_id}`;
      const what = ch.value === false ? "убран" : ch.value === "actuated" ? "адаптивный" : "добавлен";
      return `${where}: светофор ${what}`;
    }
  }
}

export const sameChanges = (a: ChangeIn[] | null | undefined, b: ChangeIn[]) =>
  JSON.stringify((a ?? []).map((c) => [c.type, c.target_id, c.value ?? null])) ===
  JSON.stringify(b.map((c) => [c.type, c.target_id, c.value ?? null]));
