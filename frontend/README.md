# Fleet health dashboard

This Next.js App Router application is the frontend for the predictive-maintenance platform.
It reads the documented FastAPI operational endpoints and never connects directly to PostgreSQL.

## Local development

From this directory:

```bash
npm install
npm run dev
```

The dashboard runs at `http://localhost:3000`. Copy `.env.example` to `.env.local` if the
FastAPI service is not available at `http://127.0.0.1:8000`.
