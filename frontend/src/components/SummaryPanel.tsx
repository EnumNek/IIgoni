// Сводка по сети: «до» (базовая сеть, обычные светофоры) против «после» (сценарий, выбранный режим).
import { duration, edgeTitle, pct } from "../network";
import type { Compare, Forecast, Network, Summary } from "../types";

interface Row {
  key: keyof Summary;
  label: string;
  fmt: (v: number | null | undefined) => string;
  better: "up" | "down";
}

const ROWS: Row[] = [
  { key: "mean_speed", label: "Средняя скорость", fmt: (v) => (v == null ? "—" : `${(v * 3.6).toFixed(1)} км/ч`), better: "up" },
  { key: "mean_trip_time", label: "Среднее время в пути", fmt: duration, better: "down" },
  { key: "mean_occupancy", label: "Средняя загруженность", fmt: (v) => pct(v, 1), better: "down" },
  { key: "congested_share", label: "Перегружено (по длине)", fmt: (v) => pct(v, 1), better: "down" },
];

interface Props {
  baseForecast: Forecast | null;
  forecast: Forecast | null;
  comparison: Compare | null;
  network: Network | null;
  onFocus: (edgeId: string) => void;
}

export default function SummaryPanel({ baseForecast, forecast, comparison, network, onFocus }: Props) {
  const a = baseForecast?.summary;
  const b = comparison ? forecast?.summary : null;
  const modeLabel = (m?: string) => (m === "rl" ? "ИИ-агенты" : m === "actuated" ? "адаптивные" : "обычные");
  const warnings = [...new Set([...(baseForecast?.summary?.warnings ?? []), ...(forecast?.summary?.warnings ?? [])])];
  const edgesById = new Map(network?.edges.map((e) => [e.id, e]));

  return (
    <section className="panel">
      <h2>Сводка по району</h2>
      {!a ? (
        <p className="hint">Нет данных прогноза.</p>
      ) : (
        <table className="summary">
          <thead>
            <tr>
              <th />
              <th>До</th>
              {b && <th>После</th>}
              {b && <th>Δ</th>}
            </tr>
            <tr className="sub">
              <th />
              <th>база, {modeLabel(baseForecast?.signal_mode)}</th>
              {b && <th>{modeLabel(forecast?.signal_mode)}</th>}
              {b && <th />}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => {
              const va = a[r.key] as number | null | undefined;
              const vb = b?.[r.key] as number | null | undefined;
              const d = comparison?.delta[r.key];
              const good = d ? (r.better === "up" ? d.abs > 0 : d.abs < 0) : false;
              const tiny = d ? Math.abs(d.pct ?? 0) < 0.5 : true;
              return (
                <tr key={r.key}>
                  <th>{r.label}</th>
                  <td>{r.fmt(va)}</td>
                  {b && <td>{r.fmt(vb)}</td>}
                  {b && (
                    <td className={tiny ? "muted" : good ? "good" : "bad"}>
                      {d?.pct != null ? `${d.pct > 0 ? "+" : ""}${d.pct.toFixed(1)} %` : "—"}
                    </td>
                  )}
                </tr>
              );
            })}
            {b && (b.closed_count ?? 0) > 0 && (
              <tr>
                <th>Закрыто участков</th>
                <td>{a.closed_count ?? 0}</td>
                <td>{b.closed_count}</td>
                <td />
              </tr>
            )}
          </tbody>
        </table>
      )}

      {comparison && comparison.top_worsened.length > 0 && (
        <details className="top">
          <summary>Где станет хуже ({comparison.top_worsened.length})</summary>
          <ul>
            {comparison.top_worsened.slice(0, 6).map((t) => (
              <li key={t.edge_id}>
                <button className="link" onClick={() => onFocus(t.edge_id)}>{edgeTitle(edgesById.get(t.edge_id))}</button>
                <span className="bad"> +{duration(t.d_travel_time)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {comparison && comparison.top_improved.length > 0 && (
        <details className="top">
          <summary>Где станет лучше ({comparison.top_improved.length})</summary>
          <ul>
            {comparison.top_improved.slice(0, 6).map((t) => (
              <li key={t.edge_id}>
                <button className="link" onClick={() => onFocus(t.edge_id)}>{edgeTitle(edgesById.get(t.edge_id))}</button>
                <span className="good"> −{duration(-t.d_travel_time)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {warnings.length > 0 && (
        <details className="top">
          <summary className="muted">Предупреждения ({warnings.length})</summary>
          <ul className="small muted">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
