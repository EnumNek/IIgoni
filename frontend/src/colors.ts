// Цветовые шкалы карты. Пороги загруженности (occupancy — доля длины участка, занятая автомобилями,
// как в SUMO edgeData): при плотном потоке ≈ 0,3, «пробка вплотную» — 0,6 и выше.
import type { ExpressionSpecification } from "maplibre-gl";

export const OCC_STOPS: [number, string][] = [
  [0.0, "#1a9850"],
  [0.08, "#91cf60"],
  [0.16, "#fee08b"],
  [0.28, "#fc8d59"],
  [0.42, "#d73027"],
  [0.6, "#7f0000"],
];

// Относительное изменение времени проезда (после/до − 1): синее — стало лучше, красное — хуже
export const DIFF_STOPS: [number, string][] = [
  [-0.3, "#2166ac"],
  [-0.1, "#67a9cf"],
  [-0.02, "#cfd8e3"],
  [0.02, "#cfd8e3"],
  [0.1, "#ef8a62"],
  [0.3, "#b2182b"],
];

export const NO_DATA = "#8a94a6";

function interpolate(prop: string, stops: [number, string][]): ExpressionSpecification {
  return ["interpolate", ["linear"], ["get", prop], ...stops.flat()] as unknown as ExpressionSpecification;
}

export const occupancyColor: ExpressionSpecification = [
  "case",
  ["<", ["get", "occ"], 0],
  NO_DATA,
  interpolate("occ", OCC_STOPS),
] as unknown as ExpressionSpecification;

export const diffColor: ExpressionSpecification = [
  "case",
  ["!", ["has", "rel"]],
  NO_DATA,
  interpolate("rel", DIFF_STOPS),
] as unknown as ExpressionSpecification;

export function occColorJs(x: number): string {
  for (let i = OCC_STOPS.length - 1; i >= 0; i--) if (x >= OCC_STOPS[i][0]) return OCC_STOPS[i][1];
  return OCC_STOPS[0][1];
}
