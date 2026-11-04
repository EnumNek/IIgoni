// Карточка выбранного участка или перекрёстка: параметры, прогноз и действия.
import { useEffect, useState } from "react";
import { api } from "../api";
import { duration, edgeTitle, kmh, pct, reverseEdge } from "../network";
import type { ChangeIn, Forecast, Measurement, NetEdge, NetNode, Network, Selection } from "../types";

// Диапазон числа полос — как в контракте изменений sim/changes.py и backend (1..6)
const MAX_LANES = 6;

interface Props {
  selection: NonNullable<Selection>;
  baseNetwork: Network; // сеть без изменений
  network: Network; // сеть с изменениями сценария
  baseForecast: Forecast | null;
  forecast: Forecast | null; // актуальный прогноз сценария (null — нет или устарел)
  readOnlyHint: boolean;
  busy: boolean;
  onApply: (changes: ChangeIn[]) => void;
  onClose: () => void;
}

export default function SelectionPanel(p: Props) {
  if (p.selection.kind === "edge") {
    const edge = p.network.edges.find((e) => e.id === p.selection.id);
    const base = p.baseNetwork.edges.find((e) => e.id === p.selection.id);
    if (!edge || !base) return null;
    return <EdgeCard {...p} edge={edge} base={base} />;
  }
  const node = p.network.nodes.find((n) => n.id === p.selection.id);
  if (!node) return null;
  return <NodeCard {...p} node={node} />;
}

function Card({ title, subtitle, onClose, children }: { title: string; subtitle?: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="card selection" role="dialog" aria-label={title}>
      <div className="card-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <div className="muted small">{subtitle}</div>}
        </div>
        <button className="ghost" onClick={onClose} aria-label="Закрыть">
          ×
        </button>
      </div>
      {children}
    </div>
  );
}

function EdgeCard(p: Props & { edge: NetEdge; base: NetEdge }) {
  const { edge, base } = p;
  const reverse = reverseEdge(p.network, edge);
  const [lanes, setLanes] = useState(edge.lanes);
  const [speedKmh, setSpeedKmh] = useState(Math.round(edge.speed * 3.6));
  const [both, setBoth] = useState(true);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  useEffect(() => {
    setLanes(edge.lanes);
    setSpeedKmh(Math.round(edge.speed * 3.6));
  }, [edge.id, edge.lanes, edge.speed]);
  useEffect(() => {
    let alive = true;
    api.edge(edge.id).then((d) => alive && setMeasurements(d.measurements)).catch(() => alive && setMeasurements([]));
    return () => {
      alive = false;
    };
  }, [edge.id]);
  const lastM = measurements[0];

  const targets = both && reverse ? [edge.id, reverse.id] : [edge.id];
  const before = p.baseForecast?.results[edge.id];
  const after = p.forecast?.results[edge.id];
  const kmhSpeeds = [20, 30, 40, 50, 60, 70, 80];

  return (
    <Card title={edgeTitle(edge)} subtitle={`${edge.id} · ${(edge.type ?? "").replace("highway.", "")} · ${Math.round(edge.length)} м`} onClose={p.onClose}>
      <table className="kv">
        <tbody>
          <tr>
            <th>Полос</th>
            <td>{edge.lanes}{edge.lanes !== base.lanes && <span className="was"> (было {base.lanes})</span>}</td>
          </tr>
          <tr>
            <th>Ограничение</th>
            <td>{kmh(edge.speed)}{edge.speed !== base.speed && <span className="was"> (было {kmh(base.speed)})</span>}</td>
          </tr>
          <tr>
            <th>Состояние</th>
            <td>{edge.closed ? <b className="bad">закрыт</b> : "открыт"}</td>
          </tr>
        </tbody>
      </table>

      <table className="cmp">
        <thead>
          <tr>
            <th />
            <th>До</th>
            <th>После</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Загруженность</th>
            <td>{pct(before?.occupancy)}</td>
            <td>{after ? (after.closed ? "—" : pct(after.occupancy)) : "?"}</td>
          </tr>
          <tr>
            <th>Ср. скорость</th>
            <td>{kmh(before?.mean_speed)}</td>
            <td>{after ? (after.closed ? "—" : kmh(after.mean_speed)) : "?"}</td>
          </tr>
          <tr>
            <th>Время проезда</th>
            <td>{duration(before?.travel_time)}</td>
            <td>{after ? duration(after.travel_time) : "?"}</td>
          </tr>
        </tbody>
      </table>
      {!p.forecast && <p className="hint small">«?» — нет актуального прогноза, нажмите «Рассчитать».</p>}
      {lastM && (
        <p className="small muted">
          Замер ({lastM.source}{lastM.run_label ? `, ${lastM.run_label}` : ""}): загруженность {pct(lastM.occupancy)}, скорость{" "}
          {kmh(lastM.mean_speed)}
          {lastM.flow != null && `, поток ${Math.round(lastM.flow)} авт/ч`}
        </p>
      )}

      {reverse && (
        <label className="check">
          <input type="checkbox" checked={both} onChange={(e) => setBoth(e.target.checked)} /> Оба направления улицы
        </label>
      )}

      <div className="editor">
        <label>Полосы</label>
        <div className="stepper">
          <button onClick={() => setLanes(Math.max(1, lanes - 1))} aria-label="Меньше полос">−</button>
          <span>{lanes}</span>
          <button onClick={() => setLanes(Math.min(MAX_LANES, lanes + 1))} aria-label="Больше полос">+</button>
        </div>
        <button disabled={p.busy || lanes === edge.lanes} onClick={() => p.onApply(targets.map((id) => ({ type: "lanes", target_id: id, value: lanes })))}>
          Применить
        </button>

        <label htmlFor="speed">Скорость</label>
        <select id="speed" value={speedKmh} onChange={(e) => setSpeedKmh(Number(e.target.value))}>
          {[...new Set([...kmhSpeeds, Math.round(edge.speed * 3.6)])].sort((a, b) => a - b).map((s) => (
            <option key={s} value={s}>{s} км/ч</option>
          ))}
        </select>
        <button
          disabled={p.busy || speedKmh === Math.round(edge.speed * 3.6)}
          onClick={() => p.onApply(targets.map((id) => ({ type: "speed", target_id: id, value: +(speedKmh / 3.6).toFixed(3) })))}
        >
          Применить
        </button>
      </div>

      <button
        className={edge.closed ? "wide" : "wide danger"}
        disabled={p.busy}
        onClick={() => p.onApply(targets.map((id) => ({ type: edge.closed ? "open" : "close", target_id: id, value: null })))}
      >
        {edge.closed ? "Открыть участок" : "Закрыть участок"}
      </button>
      {p.readOnlyHint && <p className="hint small">Изменение создаст новый сценарий.</p>}
    </Card>
  );
}

function NodeCard(p: Props & { node: NetNode }) {
  const { node } = p;
  const incoming = p.network.edges.filter((e) => e.to === node.id && !e.closed);
  const isTls = node.type === "traffic_light";
  const factor = p.forecast?.agent_metrics?.tls_factors?.[node.id];
  const names = [...new Set(incoming.map((e) => e.name).filter(Boolean))];
  const label = isTls ? "Светофор" : node.type === "dead_end" ? "Тупик / граница района" : "Нерегулируемый перекрёсток";
  return (
    <Card title={label} subtitle={`узел ${node.id}`} onClose={p.onClose}>
      <p className="small">
        Подходов: <b>{incoming.length}</b>
        {names.length > 0 && <> · {names.join(", ")}</>}
      </p>
      {factor != null && (
        <p className="small">
          ИИ-агент меняет задержку на подходах: <b className={factor < 1 ? "good" : "bad"}>×{factor.toFixed(2)}</b>
        </p>
      )}
      <button
        className="wide primary"
        disabled={p.busy || node.type === "dead_end"}
        onClick={() => p.onApply([{ type: "traffic_light", target_id: node.id, value: !isTls }])}
      >
        {isTls ? "Убрать светофор" : "Добавить светофор"}
      </button>
      {p.readOnlyHint && <p className="hint small">Изменение создаст новый сценарий.</p>}
    </Card>
  );
}
