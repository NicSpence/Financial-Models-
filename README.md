# Financial-Models

A suite of tools and scripts for financial data and analysis.

## Architecture

* `Jupyter` for exploratory analysis and a common `Python` environment
* `MySQL` relational database for data storage
* `Celery` and `Flower` for scaling data engineering and task automation using
`Redis` as the primary broker and database backend

## Requirements

* [Docker](https://www.docker.com/)
* [docker-compose](https://docs.docker.com/compose/)

## Setup

* Copy `.env-copy` to `.env` and edit as needed

## Building

```shell
docker compose build --pull
```

## Running

```shell
docker compose up -d --build
```

Check the status with `docker ps`

View the following pages in your favorite web-browser:

* [Jupyter - http://localhost:8888](http://localhost:8888)
* [Celery via Flower - http://localhost:5555](http://localhost:5555)

Connect via your favorite SQL tool to MySQL

```shell
localhost:3306
```

TODO: add Squirrel instructions, possibly phpmyadmin?

## Debugging

```shell
docker compose logs -f
```

## Teardown

```shell
docker compose down -v
```

## Sites for data collection

https://www.nasdaq.com/market-activity/stocks/screener

https://go.factset.com/marketplace/catalog?select-one-filters=eyJwcm9kdWN0LWNhdGVnb3J5IjoiQVBJIn0%3D

https://www.morningstar.com/stocks

https://www.google.com/finance/

https://www.sec.gov/edgar

Metals

https://www.kitco.com/market/

Economics and data

https://fred.stlouisfed.org/

https://www.census.gov/economic-indicators/

https://data.worldbank.org/

## Game Plan

### Basic Skills

* Utilize paper trading and methodologies in "paper" or "dev" enivronment prior to live trades.
  
Risk assessement and tolerance

Develop macro bais (Market Ideology)

Pick the market for the trading strategy (Known sector i.e technology vs precious metals)

Develop time frame strategy (scalper vs day trader vs buy and hold etc.)

Defining goals

### Establishing Rules and Checklist

* Entry rules- Timing, technical analysis, plan of entry into trade
* Exit rules- when to take profits, trailing stops, limits etc.
* Check list- reviewing analysis prior to engaging
* DOCUMENT ALL RULES. BACKTEST ALL STRATEGIES.  MONITOR AND TRACK ALL PROGRESS
