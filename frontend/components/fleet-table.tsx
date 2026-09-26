"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { FleetEngine, HealthStatus } from "@/lib/api";

type SortKey = "engine" | "cycle" | "rul";

const statusLabels: Record<HealthStatus, string> = {
  current: "Current", stale: "Stale", degraded: "Degraded", withheld: "Withheld", unavailable: "Unavailable",
};

function StatusBadge({ status }: { status: HealthStatus }) {
  return <span className={`status-badge status-${status}`}>{statusLabels[status]}</span>;
}

export function FleetTable({ engines }: { engines: FleetEngine[] }) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<HealthStatus | "all">("all");
  const [alertsOnly, setAlertsOnly] = useState(false);
  const [sort, setSort] = useState<SortKey>("engine");

  const filteredEngines = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return engines
      .filter((engine) => !normalizedSearch || String(engine.engine_id).includes(normalizedSearch))
      .filter((engine) => status === "all" || engine.health_status === status)
      .filter((engine) => !alertsOnly || engine.open_alert_count > 0)
      .sort((left, right) => {
        if (sort === "cycle") return (right.latest_received_cycle ?? -1) - (left.latest_received_cycle ?? -1);
        if (sort === "rul") return (left.latest_prediction?.estimated_rul ?? Number.POSITIVE_INFINITY) - (right.latest_prediction?.estimated_rul ?? Number.POSITIVE_INFINITY);
        return left.engine_id - right.engine_id;
      });
  }, [alertsOnly, engines, search, sort, status]);

  return (
    <>
      <div className="table-controls" aria-label="Fleet table filters">
        <label>Search engine<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="e.g. 960022" /></label>
        <label>Health status<select value={status} onChange={(event) => setStatus(event.target.value as HealthStatus | "all")}><option value="all">All statuses</option>{Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label>Sort by<select value={sort} onChange={(event) => setSort(event.target.value as SortKey)}><option value="engine">Engine ID</option><option value="cycle">Latest cycle</option><option value="rul">Lowest RUL first</option></select></label>
        <label className="checkbox-label"><input type="checkbox" checked={alertsOnly} onChange={(event) => setAlertsOnly(event.target.checked)} /> Open alerts only</label>
      </div>
      <div className="table-result-count">Showing {filteredEngines.length} of {engines.length} engines</div>
      <div className="table-scroll"><table><thead><tr><th>Equipment</th><th>Health</th><th>Latest cycle</th><th>Estimated RUL</th><th>Open alerts</th><th>Quality issues</th></tr></thead><tbody>{filteredEngines.map((engine) => <tr key={engine.engine_id}><td className="engine-id"><Link href={`/equipment/${engine.engine_id}`}>Engine {engine.engine_id}</Link></td><td><StatusBadge status={engine.health_status} /></td><td>{engine.latest_received_cycle ?? "—"}</td><td>{engine.latest_prediction?.estimated_rul != null ? `${engine.latest_prediction.estimated_rul.toFixed(1)} cycles` : "—"}</td><td>{engine.open_alert_count}</td><td>{engine.open_data_quality_issue_count}</td></tr>)}</tbody></table></div>
      {filteredEngines.length === 0 && <div className="table-empty"><strong>No engines match these filters.</strong><span>Change the search or clear the selected filters.</span></div>}
    </>
  );
}
