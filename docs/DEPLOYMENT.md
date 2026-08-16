# Deployment Guide

## Scope

This repository ships a Streamlit dashboard. It is not a static Vite/React site
and has no Vercel deployment target, static HTML source page, sitemap, or
robots file to maintain. Do not add web-SEO files solely to satisfy a static
site checklist; they would not control Streamlit's application shell.

The app already sets a page title, favicon, and wide responsive layout in
`app/streamlit_app.py`. Streamlit page navigation supplies distinct titles for
the dashboard pages.

## Host configuration

Choose a Streamlit-compatible host. Configure it to:

1. Install dependencies from `requirements.txt` using Python 3.12 or later.
2. Start `app/streamlit_app.py`.
3. Allow outbound access only if the host is expected to run `make data`.
4. Keep provider credentials, private portfolio inputs, and generated reports
   out of the deployment repository and host logs.

The tracked synthetic dataset is the default no-credential demonstration mode.
Use a host-local `MRRP_PRICES_PATH` or `MRRP_PROCESSED_DIR` only for a reviewed
data source. Never commit those generated market-data caches.

## Pre-launch checks

Before making a deployment public:

1. Run `make check` on the exact commit to deploy.
2. Open every dashboard page at desktop and narrow mobile widths; ensure tables
   and charts remain usable and no errors appear in the browser console.
3. Confirm titles match the current page, the tab shows the chart favicon, and
   the application uses only current, clearly labelled data.
4. Verify no credentials, private portfolio data, or generated reports are in
   the deployed artifact or repository history.
5. Configure the host's canonical public URL, error handling, access policy,
   and observability according to that host's controls.

## Web-search checklist mapping

Static-site items such as `index.html` metadata, canonical tags, Open Graph
images, JSON-LD, `llms.txt`, `robots.txt`, `sitemap.xml`, JavaScript source
maps, Vite bundles, and a static 404 page do not apply to this Streamlit
repository. If the project is later rebuilt as a marketing or documentation
site, add those concerns to that separate site rather than injecting unsupported
HTML into the dashboard.
