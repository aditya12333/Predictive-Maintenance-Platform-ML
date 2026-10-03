"use client";

import { useEffect, useRef, useState } from "react";
import { sendTelemetry } from "@/lib/api";

const DEMO_INTERVAL_MS = 5_000;

function measurements(): Record<string, number> {
  return Object.fromEntries(Array.from({ length: 21 }, (_, index) => [`sensor_${index + 1}`, 1.0]));
}

export function DemoTelemetry() {
  const [engineId, setEngineId] = useState("990500");
  const [cycle, setCycle] = useState(0);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("Send sequential synthetic telemetry for a live demo.");
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);
  const cycleRef = useRef(0);

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
  }, []);

  async function sendCycle() {
    const parsedEngineId = Number(engineId);
    if (!Number.isInteger(parsedEngineId) || parsedEngineId <= 0) {
      setError("Enter a positive numeric engine ID.");
      return false;
    }
    const nextCycle = cycleRef.current + 1;
    cycleRef.current = nextCycle;
    setCycle(nextCycle);
    try {
      await sendTelemetry({
        event_id: `dashboard-demo-${parsedEngineId}-${Date.now()}-${nextCycle}`,
        engine_id: parsedEngineId,
        cycle: nextCycle,
        event_timestamp: new Date().toISOString(),
        schema_version: "telemetry-v1",
        source_id: "dashboard-demo",
        measurements: measurements(),
      });
      setMessage(`Cycle ${nextCycle} sent. The worker is processing Engine ${parsedEngineId}.`);
      setError(null);
      return true;
    } catch (sendError) {
      cycleRef.current = nextCycle - 1;
      setCycle(nextCycle - 1);
      setError(sendError instanceof Error ? sendError.message : "Telemetry request failed.");
      return false;
    }
  }

  async function start() {
    if (running) return;
    cycleRef.current = 0;
    setCycle(0);
    setRunning(true);
    setMessage("Starting demo telemetry…");
    const sent = await sendCycle();
    if (!sent) {
      setRunning(false);
      return;
    }
    timerRef.current = window.setInterval(() => void sendCycle(), DEMO_INTERVAL_MS);
  }

  function stop() {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    setRunning(false);
    setMessage(`Demo telemetry stopped after ${cycleRef.current} cycle${cycleRef.current === 1 ? "" : "s"}.`);
  }

  return (
    <section className="demo-panel" aria-label="Demo telemetry">
      <div>
        <p className="eyebrow">Demo controls</p>
        <h2>Stream test telemetry</h2>
        <p className="demo-copy">Generate valid sensor events and watch the deployed worker score them.</p>
      </div>
      <div className="demo-controls">
        <label>Engine ID<input value={engineId} onChange={(event) => setEngineId(event.target.value)} disabled={running} inputMode="numeric" /></label>
        {!running ? <button className="button" onClick={() => void start()}>Start demo stream</button> : <button className="button button-secondary" onClick={stop}>Stop stream</button>}
      </div>
      <p className={error ? "demo-status demo-error" : "demo-status"} role="status">{error ?? message}</p>
      {running && <p className="demo-hint">Sending one cycle every 5 seconds. Keep the stream running to maintain a current status.</p>}
    </section>
  );
}
