## Task 1 Dev Portal Production Deployment

This is the recommended long-lived deployment model for the Task 1 dev submission portal:

- GitHub Pages continues to host the official website
- this server runs the FastAPI portal
- `nginx` terminates HTTPS and reverse proxies to local `uvicorn`
- `systemd` keeps the portal alive across reboot

### Recommended URL

Use a dedicated subdomain, for example:

- `https://task1-dev.example.org/task1/dev`
- `https://task1-dev.example.org/task1/dev/submit`
- `https://task1-dev.example.org/task1/dev/leaderboard`

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

Edit these values before starting the service:

- `User=`
- `Group=`
- `WorkingDirectory=`
- `ExecStart=`

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now task1-dev-portal
sudo systemctl status task1-dev-portal
```

### 4. Install The nginx Site

Copy the nginx config:

```bash
sudo cp task1_dev_leaderboard/deploy/task1-dev-portal.nginx.conf /etc/nginx/sites-available/task1-dev-portal
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
curl -I http://127.0.0.1:8091/health
curl -I https://task1-dev.example.org/task1/dev
curl -I https://task1-dev.example.org/task1/dev/submit
curl -I https://task1-dev.example.org/task1/dev/leaderboard
```

### 7. Update The Official Website Links

Once the stable domain is live, update `docs/index.html` so the Task 1 dev buttons point to the deployed domain instead of local or temporary URLs.

### Operational Notes

- Keep `task1_dev_leaderboard/private/` on the server only
- keep `task1_dev_leaderboard/data/accounting_CLEF/` on the server only
- do not expose `task1_dev_leaderboard/outputs/` directly as a static directory
- do not commit participant submissions or gold files
- the app already exposes `/health` for simple monitoring
