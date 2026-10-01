import type { MonitoringReport } from "@/lib/api";

function formatCheckName(name: string): string {
  return name.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

export function MonitoringPanel({ report }: { report: MonitoringReport | null }) {
  if (!report) {
    return <section className="monitoring-panel"><p className="eyebrow">Model operations</p><h2>Monitoring report unavailable</h2><p className="muted-copy">Run the scheduled monitoring job to publish the first report.</p></section>;
  }

  return (
    <section className="monitoring-panel" aria-label="Monitoring report">
      <div className="section-heading"><div><p className="eyebrow">Model operations</p><h2>Monitoring checks</h2></div><p className="timestamp">Generated {new Date(report.generated_at).toLocaleTimeString()}</p></div>
      <div className="monitoring-checks">
        {report.checks.map((check) => (
          <div className="monitoring-check" key={check.name}>
            <div><strong>{formatCheckName(check.name)}</strong><span className={`check-status status-${check.status}`}>{check.status.replace("_", " ")}</span></div>
            <span className="monitoring-value">{check.status === "no_data" ? "—" : `${(check.value * 100).toFixed(1)}%`}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
