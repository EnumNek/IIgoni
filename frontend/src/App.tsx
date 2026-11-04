// Главный компонент: загрузка сети и сценариев, редактирование, расчёт прогноза, сравнение «до/после».
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import AgentsPanel from "./components/AgentsPanel";
import Legend from "./components/Legend";
import MapView, { type EdgeValue } from "./components/MapView";
import ScenarioPanel from "./components/ScenarioPanel";
import SelectionPanel from "./components/SelectionPanel";
import SummaryPanel from "./components/SummaryPanel";
import { applyChanges, sameChanges } from "./network";
import type {
  ChangeIn,
  Compare,
  EvaluateOut,
  Forecast,
  ForecastBrief,
  Network,
  Scenario,
  ScenarioBrief,
  Selection,
  SignalMode,
  ViewMode,
} from "./types";

const latest = (list: ForecastBrief[], mode?: SignalMode) =>
  [...list].reverse().find((f) => f.status === "done" && (!mode || f.signal_mode === mode));

const toContract = (sc: Scenario | null): ChangeIn[] =>
  (sc?.changes ?? []).map((c) => ({ type: c.type, target_id: c.target_id, value: c.value ?? null }));

export default function App() {
  const [network, setNetwork] = useState<Network | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioBrief[]>([]);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [baseForecast, setBaseForecast] = useState<Forecast | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [comparison, setComparison] = useState<Compare | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluateOut | null>(null);
  const [view, setView] = useState<ViewMode>("after");
  const [signalMode, setSignalMode] = useState<SignalMode>("fixed");
  const [selection, setSelection] = useState<Selection>(null);
  const [focus, setFocus] = useState<{ id: string; kind: "edge" | "node"; n: number } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBasemap, setShowBasemap] = useState(true);

  const run = useCallback(async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const openScenario = useCallback(async (id: number, base: Forecast | null) => {
    const sc = await api.scenario(id);
    setScenario(sc);
    setSelection(null);
    setEvaluation(null);
    const brief = latest(sc.forecasts);
    if (!brief) {
      setForecast(null);
      setComparison(null);
      return;
    }
    const f = base && brief.id === base.id ? base : await api.getForecast(brief.id);
    setForecast(f);
    setSignalMode(f.signal_mode === "fixed" ? "fixed" : "rl");
    setComparison(base && f.id !== base.id ? await api.compare(base.id, f.id) : null);
  }, []);

  // ------------------------------------------------------------ первичная загрузка
  useEffect(() => {
    run("Загрузка сети…", async () => {
      const [net, list] = await Promise.all([api.network(), api.scenarios()]);
      setNetwork(net);
      setScenarios(list);
      const base = list.find((s) => s.is_base) ?? list[0];
      const baseFull = await api.scenario(base.id);
      const bb = latest(baseFull.forecasts, "fixed");
      const bf = bb ? await api.getForecast(bb.id) : await api.forecast(base.id, "fixed");
      setBaseForecast(bf);
      const last = [...list].reverse().find((s) => !s.is_base);
      await openScenario(last ? last.id : base.id, bf);
    });
  }, [run, openScenario]);

  const refreshList = async () => setScenarios(await api.scenarios());

  // ------------------------------------------------------------ редактирование
  const ensureEditable = async (): Promise<Scenario> => {
    if (scenario && !scenario.is_base) return scenario;
    const n = scenarios.filter((s) => !s.is_base).length + 1;
    const sc = await api.createScenario(`Сценарий ${n}`, "Создан из базового состояния");
    setScenario(sc);
    setForecast(null);
    setComparison(null);
    return sc;
  };

  const addChanges = (changes: ChangeIn[]) =>
    run("Сохранение изменения…", async () => {
      const sc = await ensureEditable();
      setScenario(await api.addChanges(sc.id, changes));
      await refreshList();
    });

  const deleteChange = (changeId: number) =>
    run("Удаление изменения…", async () => {
      if (!scenario) return;
      setScenario(await api.deleteChange(scenario.id, changeId));
      await refreshList();
    });

  const clearChanges = () =>
    run("Удаление изменений…", async () => {
      if (!scenario) return;
      setScenario(await api.clearChanges(scenario.id));
      await refreshList();
    });

  const newScenario = () =>
    run("Создание сценария…", async () => {
      const n = scenarios.filter((s) => !s.is_base).length + 1;
      const sc = await api.createScenario(`Сценарий ${n}`);
      await refreshList();
      await openScenario(sc.id, baseForecast);
    });

  const deleteScenario = () =>
    run("Удаление сценария…", async () => {
      if (!scenario || scenario.is_base) return;
      await api.deleteScenario(scenario.id);
      const list = await api.scenarios();
      setScenarios(list);
      const base = list.find((s) => s.is_base)!;
      await openScenario(base.id, baseForecast);
    });

  const selectScenario = (id: number) => run("Загрузка сценария…", () => openScenario(id, baseForecast));

  // ------------------------------------------------------------ расчёт
  const calculate = () =>
    run(signalMode === "rl" ? "Прогноз + оценка ИИ-агентов…" : "Расчёт прогноза…", async () => {
      if (!scenario) return;
      const f = await api.forecast(scenario.id, signalMode);
      setForecast(f);
      setComparison(baseForecast && f.id !== baseForecast.id ? await api.compare(baseForecast.id, f.id) : null);
      setScenario(await api.scenario(scenario.id));
      await refreshList();
      if (view === "before") setView("after");
    });

  const evaluateAgents = () =>
    run("Оценка режимов светофоров…", async () => {
      setEvaluation(await api.evaluate(scenario?.id ?? null, ["fixed", "actuated", "rl"]));
    });

  // ------------------------------------------------------------ производные данные для карты
  const contractChanges = useMemo(() => toContract(scenario), [scenario]);
  const afterNetwork = useMemo(
    () => (network ? applyChanges(network, contractChanges) : null),
    [network, contractChanges],
  );
  const stale = !forecast || !sameChanges(forecast.changes_snapshot, contractChanges) || forecast.signal_mode !== signalMode;
  const shownForecast = forecast ?? baseForecast;
  const effectiveView: ViewMode = view === "diff" && !comparison ? "after" : view;

  const mapNetwork = effectiveView === "before" ? network : afterNetwork;
  const values = useMemo(() => {
    const out: Record<string, EdgeValue> = {};
    if (effectiveView === "diff" && comparison) {
      for (const [id, c] of Object.entries(comparison.edges)) out[id] = { rel: c.rel_travel_time, occ: c.b?.occupancy, speed: c.b?.mean_speed };
      return out;
    }
    const f = effectiveView === "before" ? baseForecast : shownForecast;
    for (const [id, r] of Object.entries(f?.results ?? {})) out[id] = { occ: r.closed ? -1 : r.occupancy, speed: r.mean_speed };
    return out;
  }, [effectiveView, comparison, baseForecast, shownForecast]);

  const changedEdges = useMemo(
    () => new Set(effectiveView === "before" ? [] : contractChanges.filter((c) => c.type !== "traffic_light").map((c) => c.target_id)),
    [contractChanges, effectiveView],
  );
  const changedNodes = useMemo(
    () => new Set(effectiveView === "before" ? [] : contractChanges.filter((c) => c.type === "traffic_light").map((c) => c.target_id)),
    [contractChanges, effectiveView],
  );

  const focusOn = (kind: "edge" | "node", id: string) => {
    setSelection({ kind, id });
    setFocus((f) => ({ kind, id, n: (f?.n ?? 0) + 1 }));
  };

  const integ = network?.meta.integrations;

  return (
    <div className="app">
      <aside className="sidebar">
        <header className="brand">
          <h1>Транспортная сеть района</h1>
          <p>Прогноз загруженности графовой нейросетью и управление светофорами ИИ-агентами</p>
        </header>

        <ScenarioPanel
          network={network}
          scenarios={scenarios}
          scenario={scenario}
          onSelect={selectScenario}
          onNew={newScenario}
          onDelete={deleteScenario}
          onDeleteChange={deleteChange}
          onClear={clearChanges}
          onFocus={focusOn}
        />

        <section className="panel">
          <h2>Светофоры</h2>
          <div className="segmented" role="radiogroup" aria-label="Режим управления светофорами">
            <button role="radio" aria-checked={signalMode === "fixed"} className={signalMode === "fixed" ? "on" : ""} onClick={() => setSignalMode("fixed")}>
              Обычные
            </button>
            <button role="radio" aria-checked={signalMode === "rl"} className={signalMode === "rl" ? "on" : ""} onClick={() => setSignalMode("rl")}>
              ИИ-агенты (RL)
            </button>
          </div>
          <p className="hint">
            {signalMode === "fixed"
              ? "Жёсткие циклы, как сейчас на улицах. Прогноз — напрямую от нейросети."
              : "Каждым светофором управляет DQN-агент; задержки на подходах корректируются по оценке агентов."}
          </p>
          <button className="primary big" disabled={!!busy || !scenario} onClick={calculate}>
            {busy && /^(Прогноз|Расчёт)/.test(busy) ? busy : "Рассчитать"}
          </button>
          {stale && forecast !== null && <p className="warn">Сценарий изменён после расчёта — нажмите «Рассчитать».</p>}
          {!forecast && scenario && !scenario.is_base && <p className="warn">Для сценария ещё нет прогноза.</p>}
        </section>

        <SummaryPanel baseForecast={baseForecast} forecast={forecast} comparison={comparison} onFocus={(id) => focusOn("edge", id)} network={network} />

        <AgentsPanel evaluation={evaluation} busy={!!busy} onEvaluate={evaluateAgents} />

        <footer className="footer">
          <div>
            Модель: {shownForecast?.model ? `${shownForecast.model.name} v${shownForecast.model.version}` : "—"}
            {shownForecast?.predictor === "stub" && <span className="badge">заглушка</span>}
          </div>
          <div>
            Граф: {String(network?.meta.source ?? "—")}, {network?.edges.length ?? 0} участков · агенты: {integ?.evaluator.source ?? "—"}
            {integ?.evaluator.source === "stub" && <span className="badge">заглушка</span>}
          </div>
          <label className="check">
            <input type="checkbox" checked={showBasemap} onChange={(e) => setShowBasemap(e.target.checked)} /> Подложка карты
          </label>
        </footer>
      </aside>

      <main className="main">
        <div className="view-toggle segmented" role="radiogroup" aria-label="Что показать на карте">
          {(
            [
              ["before", "До"],
              ["after", "После"],
              ["diff", "Разница"],
            ] as [ViewMode, string][]
          ).map(([v, label]) => (
            <button
              key={v}
              role="radio"
              aria-checked={view === v}
              className={view === v ? "on" : ""}
              disabled={v === "diff" && !comparison}
              title={v === "diff" && !comparison ? "Сначала рассчитайте сценарий с изменениями" : undefined}
              onClick={() => setView(v)}
            >
              {label}
            </button>
          ))}
        </div>

        {mapNetwork && (
          <MapView
            network={mapNetwork}
            values={values}
            colorMode={effectiveView === "diff" ? "diff" : "occupancy"}
            changedEdges={changedEdges}
            changedNodes={changedNodes}
            selection={selection}
            onSelect={setSelection}
            focus={focus}
            showBasemap={showBasemap}
          />
        )}

        <Legend mode={effectiveView === "diff" ? "diff" : "occupancy"} hasData={!!shownForecast} />

        {afterNetwork && network && selection && (
          <SelectionPanel
            selection={selection}
            baseNetwork={network}
            network={afterNetwork}
            baseForecast={baseForecast}
            forecast={stale ? null : forecast}
            readOnlyHint={!!scenario?.is_base}
            busy={!!busy}
            onApply={addChanges}
            onClose={() => setSelection(null)}
          />
        )}

        {busy && <div className="busy" role="status">{busy}</div>}
        {error && (
          <div className="error" role="alert">
            <span>{error}</span>
            <button onClick={() => setError(null)} aria-label="Закрыть">×</button>
          </div>
        )}
      </main>
    </div>
  );
}
