"use client";

import { useEffect, useState } from "react";
import { FleetTable } from "@/components/fleet-table";
import { MonitoringPanel } from "@/components/monitoring-panel";
import { DemoTelemetry } from "@/components/demo-telemetry";
import { getFleetHealth, getMonitoringReport, type FleetHealth, type HealthStatus, type MonitoringReport } from "@/lib/api";

function SummaryCard({ label, value, tone }: { label: string; value: number; tone: HealthStatus }) {
  return <article className="summary-card"><div className={`summary-icon icon-${tone}`} /><div><p>{label}</p><strong>{value}</strong></div></article>;
}

export function FleetDashboard({ initialFleet, initialMonitoring }: { initialFleet: FleetHealth; initialMonitoring: MonitoringReport | null }) {
  const [fleet, setFleet] = useState(initialFleet);
  const [lastUpdated, setLastUpdated] = useState(new Date(initialFleet.generated_at));
  const [refreshing, setRefreshing] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [monitoring, setMonitoring] = useState(initialMonitoring);

  async function refresh() {
    setRefreshing(true);
    try {
      const nextFleet = await getFleetHealth();
      const nextMonitoring = await getMonitoringReport();
      setFleet(nextFleet);
      setMonitoring(nextMonitoring);
      setLastUpdated(new Date(nextFleet.generated_at));
      setConnectionError(null);
    } catch {
      setConnectionError("The operational API could not be reached. Showing the last successful snapshot.");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    const interval = window.setInterval(() => void refresh(), 30_000);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <>
      <section className="summary-grid" aria-label="Fleet health summary">
        <SummaryCard label="Total engines" value={fleet.total_engines} tone="current" />
        <SummaryCard label="Current" value={fleet.current_engines} tone="current" />
        <SummaryCard label="Stale" value={fleet.stale_engines} tone="stale" />
        <SummaryCard label="Degraded" value={fleet.degraded_engines} tone="degraded" />
        <SummaryCard label="Withheld" value={fleet.withheld_engines} tone="withheld" />
        <SummaryCard label="Unavailable" value={fleet.unavailable_engines} tone="unavailable" />
      </section>
      <DemoTelemetry />
      <section className="table-panel">
        <div className="section-heading"><div><p className="eyebrow">Operational inventory</p><h2>Equipment overview</h2></div><div className="refresh-panel"><p className="timestamp">Updated {lastUpdated.toLocaleTimeString()}</p><button className="refresh-button" onClick={() => void refresh()} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh"}</button></div></div>
        {connectionError && <div className="connection-error" role="status">{connectionError}</div>}
        <FleetTable engines={fleet.engines} />
      </section>
      <MonitoringPanel report={monitoring} />
    </>
  );
}
