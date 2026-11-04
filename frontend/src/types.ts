// Типы данных — повторяют контракт и схемы API бэкенда (backend/app/schemas.py).

export type NodeType = "priority" | "traffic_light" | "dead_end" | string;

export interface NetNode {
  id: string;
  lon: number;
  lat: number;
  type: NodeType;
}

export interface NetEdge {
  id: string;
  from: string;
  to: string;
  length: number; // м
  lanes: number;
  speed: number; // м/с
  priority: number;
  geometry: [number, number][];
  name?: string | null;
  type?: string | null;
  closed?: boolean;
}

export interface Network {
  meta: {
    source?: string;
    bounds?: [number, number, number, number];
    integrations?: { predictor: { source: string; error?: string | null }; evaluator: { source: string; error?: string | null } };
    [k: string]: unknown;
  };
  nodes: NetNode[];
  edges: NetEdge[];
}

export type ChangeType = "lanes" | "speed" | "close" | "open" | "traffic_light";
export type SignalMode = "fixed" | "actuated" | "rl";

export interface ChangeIn {
  type: ChangeType;
  target_id: string;
  value?: number | boolean | null;
}

export interface Change extends ChangeIn {
  id: number;
  seq: number;
  created_at: string;
}

export interface Summary {
  mean_speed?: number;
  mean_trip_time?: number | null;
  mean_edge_travel_time?: number;
  total_travel_time?: number;
  mean_occupancy?: number;
  congested_share?: number;
  edges_count?: number;
  closed_count?: number;
  warnings?: string[];
}

export interface ForecastBrief {
  id: number;
  signal_mode: SignalMode;
  status: string;
  predictor: string | null;
  created_at: string;
  summary: Summary | null;
}

export interface ScenarioBrief {
  id: number;
  name: string;
  is_base: boolean;
  created_at: string;
  changes_count: number;
  forecasts_count: number;
}

export interface Scenario {
  id: number;
  name: string;
  description: string | null;
  is_base: boolean;
  created_at: string;
  updated_at: string;
  changes: Change[];
  forecasts: ForecastBrief[];
}

export interface EdgeForecast {
  occupancy: number;
  mean_speed: number;
  travel_time: number | null;
  closed: boolean;
}

export interface AgentMetrics {
  mode?: string;
  mean_waiting_time?: number;
  mean_travel_time?: number;
  mean_speed?: number;
  throughput?: number;
  mean_queue_length?: number;
  is_stub?: boolean;
  evaluator?: string;
  [k: string]: unknown;
}

export interface Forecast {
  id: number;
  scenario_id: number;
  signal_mode: SignalMode;
  status: string;
  predictor: string | null;
  model: { name: string; version: string; is_stub: boolean; architecture?: string | null } | null;
  changes_snapshot: ChangeIn[] | null;
  summary: Summary | null;
  agent_metrics: { tls_factors?: Record<string, number>; [mode: string]: unknown } | null;
  duration_ms: number | null;
  created_at: string;
  results: Record<string, EdgeForecast>;
}

export interface CompareEdge {
  a: EdgeForecast | null;
  b: EdgeForecast | null;
  d_occupancy?: number;
  d_mean_speed?: number;
  d_travel_time?: number;
  rel_travel_time?: number | null;
}

export interface Compare {
  a: { forecast_id: number; signal_mode: SignalMode; summary: Summary };
  b: { forecast_id: number; signal_mode: SignalMode; summary: Summary };
  delta: Record<string, { a: number; b: number; abs: number; pct: number | null }>;
  edges: Record<string, CompareEdge>;
  top_improved: { edge_id: string; d_travel_time: number }[];
  top_worsened: { edge_id: string; d_travel_time: number }[];
}

export interface EvaluateOut {
  scenario_id: number | null;
  config_hash: string;
  results: Record<string, AgentMetrics>;
  warnings: string[];
}

export type Selection = { kind: "edge"; id: string } | { kind: "node"; id: string } | null;
export type ViewMode = "before" | "after" | "diff";

export interface Measurement {
  id: number;
  source: string;
  run_label: string | null;
  interval_begin: number | null;
  interval_end: number | null;
  occupancy: number | null;
  mean_speed: number | null;
  travel_time: number | null;
  flow: number | null;
}

export interface EdgeDetail {
  edge: NetEdge;
  measurements: Measurement[];
}
