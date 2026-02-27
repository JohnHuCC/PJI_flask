# AWS Quickstart (EC2 + Domain + HTTPS)

This is the fastest path to publish this project for demo/review.

## 0. What you need
- A domain name (optional but recommended)
- AWS account
- Your SSH key pair (`.pem`)

## 1. Create EC2
- AMI: Ubuntu 22.04 LTS
- Instance type: `t3.medium` (or `t3.large` if model inference is heavy)
- Storage: at least 30 GB
- Security Group inbound:
  - `22` (SSH) from your IP only
  - `80` (HTTP) from `0.0.0.0/0`
  - `443` (HTTPS) from `0.0.0.0/0`
- Outbound: allow all

## 2. SSH into EC2
```bash
ssh -i <your-key>.pem ubuntu@<EC2_PUBLIC_IP>
```

## 3. Bootstrap Docker
```bash
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/<your-org>/<your-repo>/main/scripts/aws_bootstrap_ubuntu.sh)"
```

If you don't want curl-run, clone repo first and run:
```bash
git clone https://github.com/JohnHuCC/PJI_flask.git
cd PJI_flask
sudo bash scripts/aws_bootstrap_ubuntu.sh
```

Then re-login:
```bash
exit
ssh -i <your-key>.pem ubuntu@<EC2_PUBLIC_IP>
```

## 4. Prepare production config
```bash
cd PJI_flask
cp .env.prod.example .env.prod
```

Edit `.env.prod`:
- `SECRET_KEY`: long random string
- `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD`: strong passwords
- `ALLOWED_HOSTS`: your domain(s), e.g. `demo.example.com,www.demo.example.com`
- keep `ENABLE_ADMIN_ROUTES=false`

## 5. Configure domain
At your DNS provider:
- `A` record: `@` -> `<EC2_PUBLIC_IP>`
- `A` record: `www` -> `<EC2_PUBLIC_IP>`

## 6. Configure nginx domain
Edit `deployment/nginx/conf.d/pji.conf` and replace:
- `your-domain.com` -> real domain

## 7. Start production stack
```bash
bash scripts/deploy_prod.sh
```

## 8. Issue HTTPS cert
```bash
docker compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  -d <your-domain> -d www.<your-domain> \
  --email <you@example.com> --agree-tos --no-eff-email

docker compose -f docker-compose.prod.yml restart nginx
```

## 9. Verify
- `https://<your-domain>/auth_login`
- Login and open `/`, `/personal_info`, `/model_diagnosis`, `/reactive_diagram`

## 10. Operations
```bash
# status
docker compose -f docker-compose.prod.yml ps

# logs
docker compose -f docker-compose.prod.yml logs -f web

# restart
docker compose -f docker-compose.prod.yml restart
```

## Notes
- MySQL is internal-only in `docker-compose.prod.yml` (no public DB port).
- `adjust_rate_limiting` route is disabled by default in production.
