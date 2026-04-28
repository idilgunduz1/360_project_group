# IE360 Spring 2026 Project — Electricity Price Forecasting

## Project Overview
This project is developed for **IE360 Statistical Forecasting and Time Series (Spring 2026)**.

The goal is to forecast Turkey’s **hourly day-ahead electricity Market Clearing Price (MCP / PTF)** for the **next day**. The project is organized as a forecasting competition, and the final system is expected to run automatically from a GitHub repository. According to the project description, on day `d`, forecasts are required for day `d+1`, while only price information up to the end of day `d-1` is assumed to be available. The required output is therefore **24 hourly predictions** for the target day. :contentReference[oaicite:0]{index=0}

## Current Status
This repository currently contains the **local development version** prepared before the official GitHub Classroom repository, helper functions, and submission format are fully released.

At this stage, the project includes a working end-to-end forecasting pipeline that can:

- pull recent MCP/PTF price data automatically from EPİAŞ
- pull historical and forecast weather data from Open-Meteo
- generate calendar, lag, and rolling features
- train an initial baseline model
- create a 24-hour next-day prediction file
- run local rolling backtests for model comparison

This current version is intended as an **initial working prototype** for the setup phase, which is due on **May 1, 2026**. :contentReference[oaicite:1]{index=1}

---

## Project Timeline
- **Setup deadline:** May 1, 2026
- **Test phase:** April 27 – May 3, 2026
- **Competition phase:** May 4 – May 29, 2026
- **Report due:** June 11, 2026 


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
