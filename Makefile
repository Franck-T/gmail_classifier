build:
	docker compose build --no-cache app


up:
	docker compose up -d

down:
	docker compose down

check-main-file:
	docker compose exec app cat /app/app/main.py


freeze:
	docker compose run --rm app pip freeze > requirements.txt