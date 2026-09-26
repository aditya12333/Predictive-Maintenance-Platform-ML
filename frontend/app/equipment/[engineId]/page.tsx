import Link from "next/link";
import { AlertActions } from "@/components/alert-actions";
import { getEquipmentDetail, getPredictionHistory, type Alert, type HealthStatus, type Prediction, type QualityIssue } from "@/lib/api";

const statusLabels: Record<HealthStatus, string> = {
  current: "Current", stale: "Stale", degraded: "Degraded", withheld: "Withheld", unavailable: "Unavailable",
};

function StatusBadge({ status }: { status: HealthStatus }) {
  return <span className={`status-badge status-${status}`}>{statusLabels[status]}</span>;
}

function PredictionCard({ prediction, label }: { prediction: Prediction | null; label: string }) {
  return (
    <article className="detail-card">
      <p className="eyebrow">{label}</p>
      {prediction ? (
        <>
          <strong className="rul-value">{prediction.estimated_rul == null ? "—" : `${prediction.estimated_rul.toFixed(1)} cycles`}</strong>
          <p className="muted-copy">Status: <strong>{prediction.status}</strong></p>
          <p className="muted-copy">Cycle {prediction.cycle} · Model {prediction.model_release}</p>
          {prediction.quality_flags.length > 0 && <p className="quality-note">Quality flags: {prediction.quality_flags.join(", ")}</p>}
        </>
      ) : <p className="empty-copy">No prediction is available for this equipment.</p>}
    </article>
  );
}

function AlertItem({ alert }: { alert: Alert }) {
  return <li className="alert-item"><div><strong>{alert.title}</strong><p>{alert.message}</p><AlertActions alertId={alert.alert_id} initialState={alert.state} initialVersion={alert.version} /></div><span className={`severity severity-${alert.severity}`}>{alert.severity}</span></li>;
}

function QualityItem({ issue }: { issue: QualityIssue }) {
  return <li className="alert-item"><div><strong>{issue.issue_type.replaceAll("_", " ")}</strong><p>{issue.message}</p></div><span className={`severity severity-${issue.severity}`}>{issue.severity}</span></li>;
}

export default async function EquipmentPage({ params }: { params: Promise<{ engineId: string }> }) {
  const { engineId } = await params;
  const numericEngineId = Number(engineId);
  if (!Number.isInteger(numericEngineId) || numericEngineId <= 0) throw new Error("Invalid engine ID");
  const [equipment, history] = await Promise.all([
    getEquipmentDetail(numericEngineId),
    getPredictionHistory(numericEngineId),
  ]);
  return (
    <main className="page-shell">
      <Link className="back-link" href="/">← Back to fleet health</Link>
      <header className="page-header detail-header">
        <div><p className="eyebrow">Equipment detail</p><h1>Engine {equipment.engine_id}</h1><p className="subtitle">Operational evidence and prediction history for one equipment unit.</p></div>
        <StatusBadge status={equipment.health_status} />
      </header>
      <section className="detail-stats">
        <div><span>Latest received cycle</span><strong>{equipment.latest_received_cycle ?? "—"}</strong></div>
        <div><span>Latest scored cycle</span><strong>{equipment.latest_scored_cycle ?? "—"}</strong></div>
        <div><span>Active alerts</span><strong>{equipment.active_alerts.length}</strong></div>
        <div><span>Quality issues</span><strong>{equipment.open_data_quality_issues.length}</strong></div>
      </section>
      <section className="detail-grid">
        <PredictionCard prediction={equipment.latest_prediction} label="Latest prediction" />
        <PredictionCard prediction={equipment.last_valid_prediction} label="Last valid prediction" />
      </section>
      <section className="detail-grid lower-grid">
        <article className="table-panel compact-panel"><div className="section-heading"><div><p className="eyebrow">Operational attention</p><h2>Active alerts</h2></div></div>{equipment.active_alerts.length > 0 ? <ul className="alert-list">{equipment.active_alerts.map((alert) => <AlertItem key={alert.alert_id} alert={alert} />)}</ul> : <p className="empty-copy">No active alerts.</p>}</article>
        <article className="table-panel compact-panel"><div className="section-heading"><div><p className="eyebrow">Input quality</p><h2>Open issues</h2></div></div>{equipment.open_data_quality_issues.length > 0 ? <ul className="alert-list">{equipment.open_data_quality_issues.map((issue) => <QualityItem key={issue.issue_id} issue={issue} />)}</ul> : <p className="empty-copy">No open data-quality issues.</p>}</article>
      </section>
      <section className="table-panel compact-panel history-panel"><div className="section-heading"><div><p className="eyebrow">Model evidence</p><h2>Prediction history</h2></div><span className="timestamp">{history.items.length} records</span></div><div className="table-scroll"><table><thead><tr><th>Cycle</th><th>RUL</th><th>Status</th><th>Quality</th><th>Generated</th></tr></thead><tbody>{history.items.map((prediction) => <tr key={`${prediction.event_id}-${prediction.model_release}`}><td>{prediction.cycle}</td><td>{prediction.estimated_rul == null ? "—" : `${prediction.estimated_rul.toFixed(1)} cycles`}</td><td>{prediction.status}</td><td>{prediction.data_quality_status}</td><td>{new Date(prediction.generated_at).toLocaleString()}</td></tr>)}</tbody></table></div></section>
    </main>
  );
}
