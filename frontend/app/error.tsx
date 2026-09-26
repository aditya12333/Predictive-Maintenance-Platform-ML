"use client";

export default function Error({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="page-shell">
      <section className="error-panel">
        <p className="eyebrow">Fleet health unavailable</p>
        <h1>We could not load the operational API.</h1>
        <p>Check that FastAPI is running and try again.</p>
        <button className="button" onClick={() => reset()}>Try again</button>
      </section>
    </main>
  );
}
