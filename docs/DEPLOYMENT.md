# Deploying Saarthi on the Ubuntu GPU server

Architecture (everything runs on one box; Neon and Qdrant Cloud stay external):

```
Internet ── Cloudflare ── cloudflared ──► web (Caddy)  ──► /api, /uploads ──► backend (FastAPI)
                                            │                                   ├─ Neon Postgres (external)
                                            └─ React app (static)               ├─ Qdrant Cloud   (external)
                                                                                ├─ Gemini API     (primary LLM)
                                                              ollama (GPU) ◄────┘  local fallback (profile "gpu")
```

Only Cloudflare can reach the app: no ports are opened on the server. The backend
and Ollama are not published to the host at all.

## 0. Working on the server

AnyDesk works for a one-off setup, but **SSH is much easier** for everything below
(copy/paste, logs, no screen lag). If you can, run `sudo apt install openssh-server`
once through AnyDesk and use SSH from then on.

## 1. Host setup (once)

```bash
git clone https://github.com/DevaP06/Hello-Saarthi-26.git
cd Hello-Saarthi-26
bash deploy/setup-ubuntu.sh        # Docker + NVIDIA container toolkit + deploy/.env.production
```

Log out and back in, then confirm containers can see the GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## 2. Fill in `deploy/.env.production`

The setup script already generated `JWT_SECRET` and `ADMIN_SECRET_TOKEN`. Fill in the rest
(see the comments in the file): `DATABASE_URL` (Neon pooled string, `postgresql+asyncpg://…?ssl=require`),
the Gemini key (both `OPENAI_API_KEY` and `GEMINI_API_KEY`), the `QDRANT_CLOUD_*` values,
and your public hostname in `APP_FRONTEND_URL` / `CORS_ORIGINS`.

Keep `SAARTHI_ENABLE_DEMO_USERS=0`. The demo accounts have public, weak passwords.

## 3. Start locally first (no tunnel yet)

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env.production up -d --build
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend     # wait for "Application startup complete"
```

The first build takes a while (PyTorch + dependencies). On start the backend applies
all DB migrations automatically. Smoke test on the server:

```bash
curl -s http://localhost:8080/health          # {"status":"healthy",...}
curl -s http://localhost:8080/ready           # DB reachable
```

## 4. Build the knowledge-base index (once, and after the KB changes)

```bash
docker compose -f docker-compose.prod.yml exec backend python -m scripts.build_kb_gemini_index
```

The index is stored in the `saarthi_kb` volume and survives restarts and rebuilds.

## 5. Publish with Cloudflare Tunnel

1. Cloudflare dashboard → **Zero Trust → Networks → Tunnels → Create a tunnel** (Cloudflared).
2. Copy the tunnel **token** into `CLOUDFLARE_TUNNEL_TOKEN` in `deploy/.env.production`.
3. In the tunnel's **Public Hostname** tab add your hostname (e.g. `saarthi.example.com`) with
   service type **HTTP** and URL **`web:80`**.
4. Start the tunnel:

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env.production --profile tunnel up -d
```

Open `https://<your hostname>`, sign up, and test chat. For a quick demo without a domain,
run `docker run --rm --network saarthi_default cloudflare/cloudflared tunnel --url http://web:80`
and use the temporary `trycloudflare.com` address it prints (set `APP_FRONTEND_URL` and
`CORS_ORIGINS` to it).

## 6. Local GPU model (fallback)

```bash
docker compose -f docker-compose.prod.yml --env-file deploy/.env.production --profile gpu up -d ollama
docker compose -f docker-compose.prod.yml exec ollama ollama pull qwen2.5:14b
docker compose -f docker-compose.prod.yml exec ollama ollama pull nomic-embed-text
```

The `LOCAL_*` variables in `.env.production` select the local model for **Tier 1** (see
`src/utils/llm.py`). Automatic *failover* (try Gemini, switch to the local model when Gemini
errors or is rate-limited) is **not wired up yet**; today the tiers route by question
complexity, not by availability. Treat the `gpu` profile as "model is ready" until that
is implemented.

## 7. Operating it

| Task | Command |
|---|---|
| Logs | `docker compose -f docker-compose.prod.yml logs -f backend` |
| Update to latest code | `git pull && docker compose -f docker-compose.prod.yml --env-file deploy/.env.production up -d --build` |
| Restart | `docker compose -f docker-compose.prod.yml restart backend` |
| Stop everything | `docker compose -f docker-compose.prod.yml down` (volumes are kept) |
| Back up uploads + KB index | `docker run --rm -v saarthi_saarthi_uploads:/u -v saarthi_saarthi_kb:/k -v "$PWD":/b ubuntu tar czf /b/saarthi-data.tgz /u /k` |

Back up `deploy/.env.production` separately and keep it private. The database is on Neon,
which keeps its own history and backups.

## 8. Code Lab sandbox

Code that students (or the AI) run never executes in the backend. The backend sends it to
the `sandbox` service (`SANDBOX_URL`), which has **no secrets** (no env file), **no internet**
(internal network only), a read-only filesystem, runs as `nobody`, and is capped at 512 MB,
1 CPU and 128 processes. The sandbox image provides Python and GNU Octave for MATLAB-compatible
scripts, including the Octave signal package. Scripts using it can start with `pkg load signal`.

Caddy sends `PROXY_SECRET` with every request; the backend trusts the forwarded client IP
only when it is present, so code in the sandbox cannot fake its IP to dodge rate limits.
`docker compose` refuses to start without `PROXY_SECRET`.

On a developer laptop (`python main.py`, no `SANDBOX_URL`) code runs locally, with the
API's own environment: never expose that setup to other people.

## 9. Known limits

- **One API worker.** The rate limiter, semantic cache and token budget keep their state in
  process memory. Do not raise `--workers` without moving that state to Redis.
- **Uploads are on local disk** (the `saarthi_uploads` volume) and are served publicly at
  `/uploads/…` to anyone with the link. Back the volume up. Moving files to object storage
  is the longer-term fix.
- **Heavy ingestion** (OCR, Whisper, KB building from raw PDFs/videos) is not in the image. Run
  those pipelines on the host with `requirements-content-pipeline.txt` in a virtualenv;
  they write to Qdrant, and the backend only reads from it.
- Rotate any credential that has ever been pasted into chat or committed.
