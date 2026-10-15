# Deploying CustIQ to an Oracle Cloud free VM (Ampere/ARM, Linux)

Hosts **both** the API and the frontend on one always-on VM behind Nginx, with
the API managed by systemd. The repo is a monorepo: `backend/` (FastAPI) and
`frontend/` (React). Unlike free PaaS tiers, the VM never sleeps.

Assumptions: Ubuntu on an Oracle Ampere A1 (ARM64/aarch64) instance, user
`ubuntu`, your Gemini API key ready. Adjust paths/user if different.

> ARM note: everything here is pure Python + Node and installs on ARM64. The
> first `pip install` (xgboost/lightgbm/shap wheels) takes a few minutes — fine
> on 4 GB RAM.

---

## 1. Open the ports (Oracle is closed by default)

1. **VCN Security List / NSG** (Oracle console): add an ingress rule for TCP
   **80** (and **443** for TLS) from `0.0.0.0/0`.
2. **The VM's own firewall:**
   ```bash
   sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
   sudo netfilter-persistent save
   ```

## 2. Install system packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip nginx git curl
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

## 3. Clone the monorepo

```bash
cd ~
git clone https://github.com/Mat-rixMJ/custiq.git
cd custiq
```

## 4. Build and run the backend (systemd)

```bash
cd ~/custiq/backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# Downloads the real Telco dataset, then trains + tunes the model:
chmod +x build.sh && ./build.sh
```

Create `~/custiq/backend/.env` (never committed):

```bash
cat > ~/custiq/backend/.env <<'EOF'
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-real-key-here
GEMINI_MODEL=gemini-1.5-flash
CUSTIQ_CORS_ORIGINS=http://localhost:5173
EOF
chmod 600 ~/custiq/backend/.env
```

Install the service (the unit file is in `backend/deploy/`; it already points at
`~/custiq/backend`):

```bash
sudo cp ~/custiq/backend/deploy/custiq.service /etc/systemd/system/custiq.service
sudo systemctl daemon-reload
sudo systemctl enable --now custiq
systemctl status custiq --no-pager
curl -s http://127.0.0.1:8300/health      # {"status":"ok","churn_model":"RandomForest",...}
```

## 5. Build the frontend (talks to /api, same origin)

```bash
cd ~/custiq/frontend
npm install
echo 'VITE_API_BASE=/api' > .env.production
npm run build
sudo mkdir -p /var/www/custiq
sudo cp -r dist/* /var/www/custiq/
```

## 6. Nginx

```bash
sudo cp ~/custiq/backend/deploy/nginx-custiq.conf /etc/nginx/sites-available/custiq
sudo nano /etc/nginx/sites-available/custiq      # set server_name to your IP/domain
sudo ln -s /etc/nginx/sites-available/custiq /etc/nginx/sites-enabled/custiq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Visit `http://<vm-public-ip>/` — the UI loads and calls the API at `/api` on the
same origin (no CORS to configure).

## 7. (Optional) HTTPS

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## Updating later

```bash
cd ~/custiq && git pull
# backend
cd backend && ./.venv/bin/pip install -r requirements.txt && ./build.sh \
  && sudo systemctl restart custiq
# frontend
cd ../frontend && npm install && npm run build && sudo cp -r dist/* /var/www/custiq/
```

## Troubleshooting

- `journalctl -u custiq -n 50 --no-pager` — API logs.
- 502 from Nginx → API not running on 127.0.0.1:8300 (`systemctl status custiq`).
- Blank page → confirm `dist/*` is in `/var/www/custiq` and Nginx `root` points there.

## After it's live

In `portfolio/src/data/projects.js`, set the CustIQ card's `status` to `'live'`
and `liveUrl` to your VM URL. Since the VM never sleeps, drop the "~30s wake" note.
