// Легенда карты.
import { DIFF_STOPS, OCC_STOPS } from "../colors";

export default function Legend({ mode, hasData }: { mode: "occupancy" | "diff"; hasData: boolean }) {
  const stops = mode === "diff" ? DIFF_STOPS : OCC_STOPS;
  const min = stops[0][0];
  const max = stops[stops.length - 1][0];
  const gradient = stops.map(([v, c]) => `${c} ${((v - min) / (max - min)) * 100}%`).join(", ");
  return (
    <div className="legend card" aria-label="Легенда">
      <div className="legend-title">{mode === "diff" ? "Изменение времени проезда" : "Загруженность участка"}</div>
      {hasData ? (
        <>
          <div className="legend-bar" style={{ background: `linear-gradient(90deg, ${gradient})` }} />
          <div className="legend-ticks">
            {mode === "diff" ? (
              <>
                <span>−30 % лучше</span>
                <span>0</span>
                <span>хуже +30 %</span>
              </>
            ) : (
              <>
                <span>свободно</span>
                <span>30 %</span>
                <span>60 %+ пробка</span>
              </>
            )}
          </div>
        </>
      ) : (
        <div className="small muted">нет прогноза</div>
      )}
      <div className="legend-symbols">
        <span><i className="sym closed" /> закрыт</span>
        <span><i className="sym changed" /> изменён</span>
        <span><i className="sym tls" /> светофор</span>
      </div>
    </div>
  );
}
