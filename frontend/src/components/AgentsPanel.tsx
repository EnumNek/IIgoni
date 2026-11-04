// Сравнение режимов светофорного управления: обычные / адаптивные / ИИ-агенты (POST /agents/evaluate).
import { duration } from "../network";
import type { EvaluateOut } from "../types";

const MODES: [string, string][] = [
  ["fixed", "Обычные"],
  ["actuated", "Адаптивные"],
  ["rl", "ИИ-агенты"],
];

interface Props {
  evaluation: EvaluateOut | null;
  busy: boolean;
  onEvaluate: () => void;
}

export default function AgentsPanel({ evaluation, busy, onEvaluate }: Props) {
  const res = evaluation?.results;
  const fixedWait = res?.fixed?.mean_waiting_time;
  const isStub = res && Object.values(res).some((m) => m.is_stub || m.evaluator === "stub");
  return (
    <section className="panel">
      <div className="row between">
        <h2>Светофоры: сравнение режимов</h2>
        {isStub && <span className="badge">заглушка</span>}
      </div>
      {!res ? (
        <>
          <p className="hint">Оценка агентами на текущем сценарии: ожидание у светофоров, время в пути, скорость.</p>
          <button disabled={busy} onClick={onEvaluate}>Сравнить режимы</button>
        </>
      ) : (
        <>
          <table className="summary">
            <thead>
              <tr>
                <th>Режим</th>
                <th>Ожидание</th>
                <th>В пути</th>
                <th>Скорость</th>
              </tr>
            </thead>
            <tbody>
              {MODES.filter(([m]) => res[m]).map(([m, label]) => {
                const r = res[m];
                const gain = fixedWait && r.mean_waiting_time != null && m !== "fixed" ? (r.mean_waiting_time / fixedWait - 1) * 100 : null;
                return (
                  <tr key={m}>
                    <th>{label}</th>
                    <td>
                      {duration(r.mean_waiting_time)}
                      {gain != null && <div className={gain < 0 ? "good small" : "bad small"}>{gain > 0 ? "+" : ""}{gain.toFixed(0)} %</div>}
                    </td>
                    <td>{duration(r.mean_travel_time)}</td>
                    <td>{r.mean_speed != null ? `${(r.mean_speed * 3.6).toFixed(1)} км/ч` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <button className="link" disabled={busy} onClick={onEvaluate}>пересчитать</button>
          {evaluation!.warnings.length > 0 && <p className="hint small">{evaluation!.warnings.join(" ")}</p>}
        </>
      )}
    </section>
  );
}
