# Final platform architecture

- `final-architecture.png`: 2400 × 2050 static export.
- `final-architecture.gif`: 1920 × 1640, 72 frames, looping every 7.2 seconds.
- `logos/`: technology artwork and [source attribution](logos/SOURCES.md).

Regenerate from the repository root:

```bash
.venv/bin/python scripts/generate_final_architecture.py
```

The generator requires Pillow (already in the development dependencies) and
Arial or DejaVu Sans. Rendering uses the bundled PNG logos and needs no network.

The diagram distinguishes the deployed Lightsail services from the locally
verified Airflow training and approval workflow. The approved MLflow registry is
copied to AWS manually; container deployment through GitHub Actions is separate.
The retraining feedback arrow applies to the local monitoring DAG, not the AWS
cron report job. MAE monitoring needs manually supplied actual outcome labels.

Static arrows and animated markers use the same connector geometry. Rendering
checks that connector paths stay outside every component card and that card text
fits. All 18 connectors carry animated markers, including vertical paths, turns,
monitoring feedback, model loading, and deployment. Motion illustrates logical
flow; it is not a representation of production latency or an event trace.
