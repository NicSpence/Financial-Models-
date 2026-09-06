

<h1 align="center">Project Financial Model Dashboard</h1>
<h3 align="center">Predictive Modeling For Trading Stategies</h3>
<img align="right" alt="Coding" width="400" src="https://gifdb.com/images/high/stock-market-buy-money-funny-meme-0jl2wq4upqk9vcav.gif">
<h3 align="left">Project Overview:</h3>
</p> Economic Analysis
</p> Sector Analysis
</p> Technical Analysis
</p> ML Models
<p align="left">
<h3 align="left">Strategy:</h3>
<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://www.java.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="java" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://www.php.net" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/php/php-original.svg" alt="php" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://www.rust-lang.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/rust/rust-plain.svg" alt="rust" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## Architecture

* `Jupyter` for exploratory analysis and a common `Python` environment
* `MySQL` relational database for data storage
* `Celery` and `Flower` for scaling data engineering and task automation using
`Redis` as the primary broker and database backend
* `phpmyadmin` for a web-ui into `MySQL`

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
* [phpmyadmin - http://localhost:8080](http://localhost:8080)

Connect via your favorite SQL tool to MySQL

```shell
localhost:3306
```

## Debugging

```shell
docker compose logs -f
```

## Teardown

```shell
docker compose down -v
```
