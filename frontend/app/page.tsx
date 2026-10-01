import { FleetDashboard } from "@/components/fleet-dashboard";
import { getFleetHealth, getMonitoringReport } from "@/lib/api";

export default async function FleetPage() {
  const fleet = await getFleetHealth();
  let monitoring = null;
  try {
    monitoring = await getMonitoringReport();
  } catch {
    monitoring = null;
  }
  return (
    <main className="page-shell">
      <header className="page-header">
        <div><p className="eyebrow">AeroReliability Operations</p><h1>Fleet health</h1><p className="subtitle">Live equipment state from the predictive maintenance platform.</p></div>
        <span className="live-indicator"><span /> Live snapshot</span>
      </header>
      <FleetDashboard initialFleet={fleet} initialMonitoring={monitoring} />
    </main>
  );
}
