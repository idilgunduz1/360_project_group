# IE360 Spring 2026 Project – Electricity Price Forecasting

## Goal
Forecast Turkey's hourly day-ahead electricity Market Clearing Price (MCP/PTF) for the next day.

## Current Local Version
This repository currently contains a local development version prepared before the GitHub Classroom repository and official helper scripts are released.

## Project Structure
- `forecast.py`: main script
- `config.py`: paths and constants
- `src/data_loader.py`: price and weather loading
- `src/features.py`: feature engineering
- `src/model.py`: baseline models
- `src/validate.py`: local validation utilities
- `src/predict.py`: submission file generation

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt