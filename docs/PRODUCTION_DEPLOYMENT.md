# Production Deployment (VM + Nginx + HTTPS)

This guide deploys the project on a cloud VM with:
- Docker + Docker Compose
- Nginx reverse proxy
- Let's Encrypt HTTPS

## 1. Prepare VM

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Clone and prepare env

```bash
git clone https://github.com/JohnHuCC/PJI_flask.git
cd PJI_flask
cp .env.prod.example .env.prod
```

Edit `.env.prod`:
- Set strong `SECRET_KEY`
- Set strong `MYSQL_PASSWORD` and `MYSQL_ROOT_PASSWORD`
- Set `ALLOWED_HOSTS=your-domain.com,www.your-domain.com`
- Keep `ENABLE_ADMIN_ROUTES=false`

## 3. Update Nginx domain config

Edit `deployment/nginx/conf.d/pji.conf`:
- Replace `your-domain.com` with your real domain.

## 4. DNS setup

Create DNS records:
- `A` record: `@` -> VM public IP
- `A` record: `www` -> VM public IP (or CNAME to `@`)

## 5. Start stack (without HTTPS first)

```bash
docker compose -f docker-compose.prod.yml up -d --build db web nginx
```

## 6. Issue Let's Encrypt certificate

```bash
docker compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  -d your-domain.com -d www.your-domain.com \
  --email you@example.com --agree-tos --no-eff-email
```

Then reload nginx:

```bash
docker compose -f docker-compose.prod.yml restart nginx
```

## 7. Security checklist

1. Disable public DB port
- `docker-compose.prod.yml` intentionally does not expose MySQL to host.

2. Disable admin/test routes
- Keep `ENABLE_ADMIN_ROUTES=false` in `.env.prod`.

3. Replace default/test account
- Create a new admin user and rotate/remove old test credentials in DB.

## 8. Useful commands

```bash
# View service status
docker compose -f docker-compose.prod.yml ps

# View web logs
docker compose -f docker-compose.prod.yml logs -f web

# Restart services
docker compose -f docker-compose.prod.yml restart

# Stop services
docker compose -f docker-compose.prod.yml down
```
