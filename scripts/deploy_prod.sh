#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env.prod ]]; then
  echo "Missing .env.prod. Create it from .env.prod.example first."
  exit 1
fi

echo "Starting production stack..."
docker compose -f docker-compose.prod.yml pull || true
docker compose -f docker-compose.prod.yml up -d --build db web nginx

echo "Stack status:"
docker compose -f docker-compose.prod.yml ps

echo "\nNext step (HTTPS certificate):"
echo "docker compose -f docker-compose.prod.yml run --rm certbot certonly \\\" 
echo "  --webroot -w /var/www/certbot \\\" 
echo "  -d <your-domain> -d www.<your-domain> \\\" 
echo "  --email <you@example.com> --agree-tos --no-eff-email"
echo "docker compose -f docker-compose.prod.yml restart nginx"
