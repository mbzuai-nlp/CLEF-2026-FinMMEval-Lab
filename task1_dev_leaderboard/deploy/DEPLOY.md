## Task 1 Portal Production Deployment

This is the recommended long-lived deployment model for the Task 1 submission portal:

- GitHub Pages continues to host the official website
- this server runs the FastAPI portal
- `nginx` terminates HTTPS and reverse proxies to local `uvicorn`
- `systemd` keeps the portal alive across reboot

### Recommended URL

Use a dedicated subdomain for each portal mode, for example:

- `https://task1-dev.example.org/task1/dev`
- `https://task1-dev.example.org/task1/dev/submit`
- `https://task1-dev.example.org/task1/dev/leaderboard`
- `https://task1-test.example.org/task1/dev`
- `https://task1-test.example.org/task1/dev/submit`

### Portal Modes

The app supports two modes:

- `TASK1_PORTAL_MODE=dev`: shows the dev leaderboard and returns score/rank after upload.
- `TASK1_PORTAL_MODE=test`: accepts final-test submissions, validates format and coverage, and does not expose score, rank, or correct counts before the deadline.

Use `dev` for the May 6 dev leaderboard. Use `test` for the final Task 1/2 run submission window ending on May 15.

Recommended `test` mode settings:

```bash
export TASK1_PORTAL_MODE=test
export TASK1_STORAGE_BACKEND=hf_dataset
export TASK1_HF_REPO_ID=<org-or-user>/<private-dataset-repo>
export HF_TOKEN=<write-token>
```

Before publishing or deploying a public Space folder, run:

```bash
python task1_dev_leaderboard/check_public_release.py
```

### 1. DNS

Create an `A` record for your chosen subdomain and point it at this server's public IP.

Example:

- host: `task1-dev.example.org`
- value: `5.195.0.145`

Wait until DNS resolves correctly before requesting HTTPS certificates.

### 2. Python Dependencies

Install the runtime dependencies in the Python environment that will be used by `systemd`:

```bash
python3 -m pip install fastapi uvicorn python-multipart
```

### 3. Install The systemd Service

Copy the service template into `/etc/systemd/system/`:

```bash
sudo cp task1_dev_leaderboard/deploy/task1-dev-portal.service /etc/systemd/system/task1-dev-portal.service
```

For the final-test portal, use the dedicated test-mode template instead:

```bash
sudo cp task1_dev_leaderboard/deploy/task1-test-portal.service /etc/systemd/system/task1-test-portal.service
```

Create the final-test portal environment file on the server. Do not commit this file:

```bash
sudo install -m 600 /dev/null /etc/task1-test-portal.env
sudoedit /etc/task1-test-portal.env
```

Required contents:

```bash
TASK1_HF_REPO_ID=<org-or-user>/<private-dataset-repo>
HF_TOKEN=<write-token>
```

Edit these values before starting the service:

- `User=`
- `Group=`
- `WorkingDirectory=`
- `Environment=TASK1_PORTAL_MODE=`
- `Environment=TASK1_STORAGE_BACKEND=`
- `Environment=TASK1_HF_REPO_ID=`
- `ExecStart=`

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now task1-dev-portal
sudo systemctl status task1-dev-portal
```

For the final-test portal, enable `task1-test-portal` instead and verify that it listens on `127.0.0.1:8092`.

### 4. Install The nginx Site

Copy the nginx config:

```bash
sudo cp task1_dev_leaderboard/deploy/task1-dev-portal.nginx.conf /etc/nginx/sites-available/task1-dev-portal
```

For the final-test portal:

```bash
sudo cp task1_dev_leaderboard/deploy/task1-test-portal.nginx.conf /etc/nginx/sites-available/task1-test-portal
```

Edit the `server_name` in that file to your real domain, then enable it:

```bash
sudo ln -s /etc/nginx/sites-available/task1-dev-portal /etc/nginx/sites-enabled/task1-dev-portal
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Issue HTTPS Certificates

If `certbot` is installed:

```bash
sudo certbot --nginx -d task1-dev.example.org
```

If it is not installed yet, install it first using your server's package manager.

### 6. Health Checks

After deployment, verify:

```bash
curl http://127.0.0.1:8091/health
curl -I https://task1-dev.example.org/task1/dev
curl -I https://task1-dev.example.org/task1/dev/submit
curl -I https://task1-dev.example.org/task1/dev/leaderboard
```

In `test` mode, `/health` should return `"portal_mode":"test"` and `/api/task1/leaderboard` should return `404`.

### 7. Update The Official Website Links

Once the stable domain is live, update `docs/index.html` so the Task 1 dev buttons point to the deployed domain instead of local or temporary URLs.

### Operational Notes

- Keep `task1_dev_leaderboard/private/` on the server only
- keep `task1_dev_leaderboard/data/accounting_CLEF/` on the server only
- do not expose `task1_dev_leaderboard/outputs/` directly as a static directory
- do not commit participant submissions or gold files
- keep the final-test portal in `TASK1_PORTAL_MODE=test` until submissions close
- the app already exposes `/health` for simple monitoring
