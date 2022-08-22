#!/usr/bin/env bash

cat baseenv terraformenv > .env
echo "PROJECT_PATH=$(pwd)" >> .env

cp .env monitoring/.env
cp .env train/.env

cp function/.env function/tests/.env
