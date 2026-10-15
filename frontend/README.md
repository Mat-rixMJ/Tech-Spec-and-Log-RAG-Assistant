# CustIQ UI

React + Vite + Tailwind frontend for the [CustIQ API](../custiq). A three-step
flow: score a customer CSV, upload policy docs, and draft AI retention emails
grounded in those docs.

**Live demo:** _coming soon_

## Run locally

Start the backend first (see `../custiq`), then:

```bash
npm install
npm run dev      # http://localhost:5173
```

### Point at a different API

Copy `.env.example` to `.env.local` and set `VITE_API_BASE` (default
`http://127.0.0.1:8300`).

## Test & build

```bash
npm test         # format/helper unit tests (node --test)
npm run build
```

## Deploy to Netlify (free)

1. Import the repo, set base directory to `custiq-ui`.
2. Set env var `VITE_API_BASE` to your Render API URL.
3. Deploy. On the backend, add this Netlify URL to `CUSTIQ_CORS_ORIGINS`.
