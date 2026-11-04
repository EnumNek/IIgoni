// Карта района на MapLibre GL: участки окрашены по загруженности (или по изменению «до/после»),
// толщина линии зависит от числа полос, встречные направления разнесены (line-offset).
import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl, {
  type ExpressionSpecification,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type Map as MLMap,
  type StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { diffColor, occupancyColor } from "../colors";
import { kmh, pct } from "../network";
import type { Network, Selection } from "../types";

export interface EdgeValue {
  occ?: number;
  rel?: number | null;
  speed?: number;
}

interface Props {
  network: Network;
  values: Record<string, EdgeValue>;
  colorMode: "occupancy" | "diff";
  changedEdges: Set<string>;
  changedNodes: Set<string>;
  selection: Selection;
  onSelect: (s: Selection) => void;
  focus: { id: string; kind: "edge" | "node"; n: number } | null;
  showBasemap: boolean;
}

const BASEMAP: StyleSpecification = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: ["a", "b", "c", "d"].map((s) => `https://${s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png`),
      tileSize: 256,
      maxzoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> © <a href="https://carto.com/">CARTO</a>',
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#eef1f4" } },
    { id: "basemap", type: "raster", source: "basemap", paint: { "raster-saturation": -0.5, "raster-opacity": 0.85 } },
  ],
};

// Толщина линии ∝ числу полос и растёт с масштабом. В MapLibre ["zoom"] допускается только во внешнем
// interpolate, поэтому «толщина + запас» для подсветки собирается отдельной функцией.
const lanesW = (k: number) => ["*", k, ["+", 0.6, ["*", 0.6, ["get", "lanes"]]]];
const width = (extra = 0) =>
  ["interpolate", ["exponential", 1.5], ["zoom"],
    12, ["+", extra * 0.5, lanesW(0.9)], 15, ["+", extra, lanesW(2.2)], 18, ["+", extra * 1.5, lanesW(6)]] as unknown as ExpressionSpecification;
const widthExpr = width();
const offsetAt = (k: number) => ["case", ["get", "twoway"], ["+", 0.4, lanesW(k / 2)], 0];
const offsetExpr = ["interpolate", ["exponential", 1.5], ["zoom"], 12, offsetAt(0.9), 15, offsetAt(2.2), 18, offsetAt(6)] as unknown as ExpressionSpecification;

interface Hover {
  x: number;
  y: number;
  title: string;
  lines: string[];
}

export default function MapView(props: Props) {
  const { network, values, colorMode, changedEdges, changedNodes, selection, onSelect, focus, showBasemap } = props;
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const [hover, setHover] = useState<Hover | null>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const valuesRef = useRef(values);
  valuesRef.current = values;
  const fitted = useRef(false);

  // ---------------------------------------------------------------- GeoJSON для источников карты
  const edgesGeo = useMemo(() => {
    const pairs = new Set(network.edges.map((e) => `${e.from}>${e.to}`));
    return {
      type: "FeatureCollection" as const,
      features: network.edges.map((e, i) => {
        const v = values[e.id];
        const properties: Record<string, unknown> = {
          id: e.id,
          name: e.name ?? "",
          lanes: e.lanes,
          speed: e.speed,
          closed: !!e.closed,
          changed: changedEdges.has(e.id),
          twoway: pairs.has(`${e.to}>${e.from}`),
          occ: v?.occ ?? -1,
        };
        if (v?.rel != null && isFinite(v.rel)) properties.rel = v.rel;
        return { type: "Feature" as const, id: i, properties, geometry: { type: "LineString" as const, coordinates: e.geometry } };
      }),
    };
  }, [network, values, changedEdges]);

  const nodesGeo = useMemo(() => {
    // Показываем только перекрёстки (≥ 3 соседних узла) и светофоры — промежуточные точки улиц не нужны
    const neighbours = new Map<string, Set<string>>();
    for (const e of network.edges) {
      if (!neighbours.has(e.from)) neighbours.set(e.from, new Set());
      if (!neighbours.has(e.to)) neighbours.set(e.to, new Set());
      neighbours.get(e.from)!.add(e.to);
      neighbours.get(e.to)!.add(e.from);
    }
    return {
      type: "FeatureCollection" as const,
      features: network.nodes
        .filter((n) => n.type === "traffic_light" || (neighbours.get(n.id)?.size ?? 0) >= 3 || changedNodes.has(n.id))
        .map((n) => ({
          type: "Feature" as const,
          properties: { id: n.id, tls: n.type === "traffic_light", changed: changedNodes.has(n.id) },
          geometry: { type: "Point" as const, coordinates: [n.lon, n.lat] },
        })),
    };
  }, [network, changedNodes]);

  // ---------------------------------------------------------------- инициализация карты
  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({ container: container.current, style: BASEMAP, center: [69.15, 54.865], zoom: 14, attributionControl: false });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    mapRef.current = map;
    // Без интернета подложка не загрузится — это не ошибка приложения, граф рисуется и без неё
    map.on("error", (ev) => {
      if ((ev as unknown as { sourceId?: string }).sourceId !== "basemap") console.error(ev.error);
    });

    map.on("load", () => {
      map.addSource("edges", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addSource("nodes", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "edges-changed", type: "line", source: "edges", filter: ["==", ["get", "changed"], true],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#7c3aed", "line-opacity": 0.35, "line-width": width(8) },
      });
      map.addLayer({
        id: "edges-main", type: "line", source: "edges", filter: ["!=", ["get", "closed"], true],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": occupancyColor, "line-width": widthExpr, "line-offset": offsetExpr },
      });
      map.addLayer({
        id: "edges-closed", type: "line", source: "edges", filter: ["==", ["get", "closed"], true],
        paint: { "line-color": "#111827", "line-width": widthExpr, "line-dasharray": [1.2, 1], "line-offset": offsetExpr },
      });
      map.addLayer({
        id: "edges-selected", type: "line", source: "edges", filter: ["==", ["get", "id"], ""],
        layout: { "line-cap": "round" },
        paint: { "line-color": "#2563eb", "line-width": width(5), "line-opacity": 0.55, "line-offset": offsetExpr },
      });
      map.addLayer({
        id: "nodes", type: "circle", source: "nodes",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, ["case", ["get", "tls"], 2.5, 1.5], 16, ["case", ["get", "tls"], 7, 4]],
          "circle-color": ["case", ["get", "tls"], "#111827", "#ffffff"],
          "circle-stroke-color": ["case", ["get", "changed"], "#7c3aed", ["get", "tls"], "#facc15", "#6b7280"],
          "circle-stroke-width": ["case", ["get", "changed"], 3, ["get", "tls"], 2, 1],
        },
      });
      map.addLayer({
        id: "nodes-selected", type: "circle", source: "nodes", filter: ["==", ["get", "id"], ""],
        paint: { "circle-radius": 11, "circle-color": "rgba(37,99,235,0.15)", "circle-stroke-color": "#2563eb", "circle-stroke-width": 2 },
      });
      setReady(true);
    });

    // Клик: сначала ищем узел (они мельче), потом участок; область поиска ±7 px
    map.on("click", (ev) => {
      const box: [maplibregl.PointLike, maplibregl.PointLike] = [[ev.point.x - 7, ev.point.y - 7], [ev.point.x + 7, ev.point.y + 7]];
      const nodes = map.queryRenderedFeatures(box, { layers: ["nodes"] });
      if (nodes.length) return onSelectRef.current({ kind: "node", id: String(nodes[0].properties.id) });
      const edges = map.queryRenderedFeatures(box, { layers: ["edges-main", "edges-closed"] });
      if (edges.length) return onSelectRef.current({ kind: "edge", id: String(edges[0].properties.id) });
      onSelectRef.current(null);
    });
    map.on("mousemove", (ev) => {
      const f = map.queryRenderedFeatures(ev.point, { layers: ["nodes", "edges-main", "edges-closed"] })[0];
      map.getCanvas().style.cursor = f ? "pointer" : "";
      if (!f) return setHover(null);
      const p = f.properties as Record<string, unknown>;
      if (f.layer.id === "nodes") {
        setHover({ x: ev.point.x, y: ev.point.y, title: p.tls ? "Перекрёсток со светофором" : "Перекрёсток", lines: ["Нажмите, чтобы изменить"] });
        return;
      }
      const v = valuesRef.current[String(p.id)];
      const lines = [`Полос: ${p.lanes}, огр. ${Math.round(Number(p.speed) * 3.6)} км/ч`];
      if (p.closed) lines.push("Закрыт для движения");
      else if (v?.occ != null && v.occ >= 0) lines.push(`Загруженность: ${pct(v.occ)}, скорость ${kmh(v.speed)}`);
      if (v?.rel != null) lines.push(`Время проезда: ${v.rel >= 0 ? "+" : ""}${(v.rel * 100).toFixed(0)} %`);
      setHover({ x: ev.point.x, y: ev.point.y, title: String(p.name || `Участок ${p.id}`), lines });
    });
    map.on("mouseout", () => setHover(null));
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // ---------------------------------------------------------------- обновление данных и стилей
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    (map.getSource("edges") as GeoJSONSource).setData(edgesGeo);
    (map.getSource("nodes") as GeoJSONSource).setData(nodesGeo);
    if (!fitted.current && network.edges.length) {
      const b = network.meta.bounds;
      const bounds: LngLatBoundsLike = b
        ? [[b[0], b[1]], [b[2], b[3]]]
        : [[Math.min(...network.nodes.map((n) => n.lon)), Math.min(...network.nodes.map((n) => n.lat))],
           [Math.max(...network.nodes.map((n) => n.lon)), Math.max(...network.nodes.map((n) => n.lat))]];
      map.fitBounds(bounds, { padding: 30, duration: 0 });
      fitted.current = true;
    }
  }, [ready, edgesGeo, nodesGeo, network]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    map.setPaintProperty("edges-main", "line-color", colorMode === "diff" ? diffColor : occupancyColor);
  }, [ready, colorMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    map.setFilter("edges-selected", ["==", ["get", "id"], selection?.kind === "edge" ? selection.id : ""]);
    map.setFilter("nodes-selected", ["==", ["get", "id"], selection?.kind === "node" ? selection.id : ""]);
  }, [ready, selection]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    map.setLayoutProperty("basemap", "visibility", showBasemap ? "visible" : "none");
  }, [ready, showBasemap]);

  // Перелёт к объекту (например, по клику на изменение в списке)
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !focus) return;
    if (focus.kind === "edge") {
      const e = network.edges.find((x) => x.id === focus.id);
      if (!e) return;
      const xs = e.geometry.map((c) => c[0]);
      const ys = e.geometry.map((c) => c[1]);
      map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 120, maxZoom: 17 });
    } else {
      const n = network.nodes.find((x) => x.id === focus.id);
      if (n) map.flyTo({ center: [n.lon, n.lat], zoom: Math.max(map.getZoom(), 16.5) });
    }
  }, [ready, focus, network]);

  return (
    <div className="map-wrap">
      <div ref={container} className="map" role="application" aria-label="Карта транспортной сети района" />
      {hover && (
        <div className="map-tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <strong>{hover.title}</strong>
          {hover.lines.map((l) => (
            <div key={l}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}
