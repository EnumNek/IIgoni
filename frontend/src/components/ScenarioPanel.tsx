// Выбор сценария и список его изменений.
import { describeChange } from "../network";
import type { Network, Scenario, ScenarioBrief } from "../types";

interface Props {
  network: Network | null;
  scenarios: ScenarioBrief[];
  scenario: Scenario | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onDelete: () => void;
  onDeleteChange: (id: number) => void;
  onClear: () => void;
  onFocus: (kind: "edge" | "node", id: string) => void;
}

export default function ScenarioPanel(p: Props) {
  const sc = p.scenario;
  return (
    <section className="panel">
      <h2>Сценарий</h2>
      <div className="row">
        <select
          aria-label="Сценарий"
          value={sc?.id ?? ""}
          onChange={(e) => p.onSelect(Number(e.target.value))}
        >
          {p.scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.is_base ? "★ " : ""}
              {s.name}
              {s.changes_count ? ` (${s.changes_count})` : ""}
            </option>
          ))}
        </select>
        <button onClick={p.onNew} title="Создать пустой сценарий">+ Новый</button>
        {sc && !sc.is_base && (
          <button className="ghost danger" onClick={p.onDelete} title="Удалить сценарий" aria-label="Удалить сценарий">
            🗑
          </button>
        )}
      </div>

      {sc?.is_base && (
        <p className="hint">
          Базовое состояние — сеть «как есть». Нажмите на участок или перекрёсток на карте: при первом изменении
          автоматически создастся новый сценарий.
        </p>
      )}

      {sc && !sc.is_base && (
        <>
          <div className="row between">
            <h3>Изменения ({sc.changes.length})</h3>
            {sc.changes.length > 0 && (
              <button className="link" onClick={p.onClear}>
                очистить
              </button>
            )}
          </div>
          {sc.changes.length === 0 ? (
            <p className="hint">Пока нет изменений. Выберите участок или перекрёсток на карте.</p>
          ) : (
            <ol className="changes">
              {sc.changes.map((c) => (
                <li key={c.id}>
                  <button
                    className="link change-text"
                    onClick={() => p.onFocus(c.type === "traffic_light" ? "node" : "edge", c.target_id)}
                    title="Показать на карте"
                  >
                    <span className={`chip chip-${c.type}`}>{CHIP[c.type]}</span>
                    {describeChange(c, p.network)}
                  </button>
                  <button className="ghost" onClick={() => p.onDeleteChange(c.id)} aria-label="Удалить изменение">
                    ×
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}

const CHIP: Record<string, string> = {
  lanes: "полосы",
  speed: "скорость",
  close: "закрыт",
  open: "открыт",
  traffic_light: "светофор",
};
