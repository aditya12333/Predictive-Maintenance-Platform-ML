export type HealthStatus = "current" | "stale" | "degraded" | "withheld" | "unavailable";

export type FleetEngine = {
  engine_id: number;
  health_status: HealthStatus;
  latest_received_cycle: number | null;
  latest_scored_cycle: number | null;
  latest_event_timestamp: string | null;
  latest_prediction: { estimated_rul: number | null; status: string } | null;
  open_alert_count: number;
  open_data_quality_issue_count: number;
};

export type FleetHealth = {
  generated_at: string;
  total_engines: number;
  current_engines: number;
  stale_engines: number;
  degraded_engines: number;
  withheld_engines: number;
  unavailable_engines: number;
  engines: FleetEngine[];
};

export type Prediction = {
  event_id: string;
  engine_id: number;
  cycle: number;
  event_timestamp: string;
  ingestion_timestamp: string;
  generated_at: string;
  estimated_rul: number | null;
  status: "available" | "degraded" | "withheld";
  data_quality_status: "valid" | "degraded" | "invalid";
  quality_flags: string[];
  model_release: string;
  feature_version: string;
};

export type Alert = {
  alert_id: number;
  engine_id: number;
  alert_type: "equipment_risk" | "data_quality";
  state: "open" | "acknowledged" | "resolved" | "dismissed";
  severity: "info" | "warning" | "critical";
  title: string;
  message: string;
  opened_at: string;
  updated_at: string;
  version: number;
};

export type QualityIssue = {
  issue_id: number;
  issue_type: string;
  severity: "info" | "warning" | "critical";
  message: string;
  detected_at: string;
};

export type EquipmentDetail = {
  engine_id: number;
  health_status: HealthStatus;
  latest_received_cycle: number | null;
  latest_scored_cycle: number | null;
  latest_event_timestamp: string | null;
  latest_prediction: Prediction | null;
  last_valid_prediction: Prediction | null;
  active_alerts: Alert[];
  open_data_quality_issues: QualityIssue[];
};

export type PredictionHistory = { engine_id: number; items: Prediction[]; next_cursor: string | null };

async function getApi<T>(path: string): Promise<T> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  const response = await fetch(`${baseUrl}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Operational API returned ${response.status}`);
  return response.json() as Promise<T>;
}

export async function getFleetHealth(): Promise<FleetHealth> {
  return getApi<FleetHealth>("/v1/fleet/health");
}

export async function getEquipmentDetail(engineId: number): Promise<EquipmentDetail> {
  return getApi<EquipmentDetail>(`/v1/equipment/${engineId}`);
}

export async function getPredictionHistory(engineId: number): Promise<PredictionHistory> {
  return getApi<PredictionHistory>(`/v1/equipment/${engineId}/predictions`);
}
