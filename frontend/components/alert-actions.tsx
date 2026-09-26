"use client";

import { useState } from "react";

type AlertState = "open" | "acknowledged" | "resolved" | "dismissed";
type Action = "acknowledge" | "resolve" | "dismiss";

type AlertActionsProps = { alertId: number; initialState: AlertState; initialVersion: number };

export function AlertActions({ alertId, initialState, initialVersion }: AlertActionsProps) {
  const [state, setState] = useState(initialState);
  const [version, setVersion] = useState(initialVersion);
  const [pending, setPending] = useState<Action | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function transition(action: Action) {
    setPending(action);
    setError(null);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
      const response = await fetch(`${baseUrl}/v1/alerts/${alertId}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action,
          expected_version: version,
          actor_id: "dashboard-user",
          reason: `Action performed from the fleet dashboard: ${action}.`,
        }),
      });
      if (!response.ok) {
        setError(response.status === 409 ? "This alert changed elsewhere. Refresh before trying again." : "The alert action failed.");
        return;
      }
      const updated = (await response.json()) as { state: AlertState; version: number };
      setState(updated.state);
      setVersion(updated.version);
    } catch {
      setError("The API could not be reached. Check that FastAPI is running.");
    } finally {
      setPending(null);
    }
  }

  if (state === "resolved" || state === "dismissed") {
    return <span className="alert-closed">{state}</span>;
  }

  return (
    <div className="alert-actions">
      {state === "open" && <button disabled={pending !== null} onClick={() => transition("acknowledge")}>{pending === "acknowledge" ? "…" : "Acknowledge"}</button>}
      <button disabled={pending !== null} onClick={() => transition("resolve")}>{pending === "resolve" ? "…" : "Resolve"}</button>
      <button disabled={pending !== null} onClick={() => transition("dismiss")}>{pending === "dismiss" ? "…" : "Dismiss"}</button>
      {error && <span className="alert-action-error" role="alert">{error}</span>}
    </div>
  );
}
