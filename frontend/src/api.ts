// Клиент REST API бэкенда. Базовый адрес: VITE_API_URL или /api (прокси Vite / nginx).
import type { ChangeIn, Compare, EdgeDetail, EvaluateOut, Forecast, Network, Scenario, ScenarioBrief, SignalMode } from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* ответ без JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  network: () => request<Network>("GET", "/network"),
  // id участков SUMO содержат «#» — обязательно кодируем
  edge: (id: string) => request<EdgeDetail>("GET", `/edges/${encodeURIComponent(id)}?limit=5`),
  scenarios: () => request<ScenarioBrief[]>("GET", "/scenarios"),
  scenario: (id: number) => request<Scenario>("GET", `/scenarios/${id}`),
  createScenario: (name: string, description?: string) =>
    request<Scenario>("POST", "/scenarios", { name, description }),
  deleteScenario: (id: number) => request<void>("DELETE", `/scenarios/${id}`),
  addChanges: (id: number, changes: ChangeIn[]) => request<Scenario>("POST", `/scenarios/${id}/changes`, changes),
  deleteChange: (id: number, changeId: number) => request<Scenario>("DELETE", `/scenarios/${id}/changes/${changeId}`),
  clearChanges: (id: number) => request<Scenario>("DELETE", `/scenarios/${id}/changes`),
  forecast: (id: number, signal_mode: SignalMode) =>
    request<Forecast>("POST", `/scenarios/${id}/forecast`, { signal_mode }),
  getForecast: (id: number) => request<Forecast>("GET", `/forecasts/${id}`),
  compare: (a: number, b: number) => request<Compare>("GET", `/compare?a=${a}&b=${b}`),
  evaluate: (scenario_id: number | null, modes: SignalMode[]) =>
    request<EvaluateOut>("POST", "/agents/evaluate", { scenario_id, modes }),
};
