# Stock Price Forecasting

Train recurrent neural networks on historical Apple (AAPL) OHLCV data to predict daily highs and next-day price movement. This repo refactors a notebook-style workflow into a small Python package you can run from the command line.


## What it does

1. **Univariate models** — SimpleRNN and LSTM predict the `High` column using a sliding window of past values.
2. **Multivariate LSTM** — Uses Open, High, RSI, and three EMAs to predict a shifted target derived from adjusted close vs. open.
3. **Evaluation** — Reports RMSE on the held-out test period and prints a short multi-step forecast for univariate models.

Training window defaults to **2016–2020**; everything after 2020 is used for testing.

## Project layout

```
Modular_Project_Code/
├── engine.py                 # Entry point: load data, train all models
├── AAPL_Stock_Data.csv       # Historical daily prices (2015+)
├── ml_pipeline/
│   ├── utils.py              # Splits, sequences, technical indicators
│   └── train.py              # RNN / LSTM / multivariate training
├── lib/
│   └── lstm_p1.ipynb         # Original exploratory notebook
├── output/                   # Saved models (created on first run)
│   ├── model_rnn.h5
│   ├── model_lstm.h5
│   └── model_mv_lstm.h5
└── requirements.txt
```

## Requirements

- Python **3.8+**
- See `requirements.txt` for pinned versions (TensorFlow 2.12, pandas, scikit-learn, etc.)
- **pandas-ta** — used for RSI and EMA features in the multivariate pipeline (install separately; see below)

## Quick start

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pandas-ta

# Run the full training pipeline
python engine.py
```

Trained Keras models are written under `output/`. Console output includes RMSE and, for RNN/LSTM, the next 25 autoregressive predictions.

## How the pipeline works

| Step | Module | Description |
|------|--------|-------------|
| Load CSV | `engine.py` | Reads `AAPL_Stock_Data.csv` with `Date` as index |
| Train/test split | `utils.train_test_split` | Train: 2016–2020; test: 2021 onward |
| Scale | `MinMaxScaler` | Fits on training `High` prices only |
| Windowing | `utils.split_sequence` | Builds `(n_steps=1, features=1)` sequences |
| Train | `train.train_rnn_model` / `train_lstm_model` | 125-unit RNN/LSTM, RMSprop, MSE |
| Multivariate | `utils.process_and_split_multivariate_data` | Adds RSI + EMAs, predicts next-day target |
| Save | `train.*` | Optional `.h5` paths passed from `engine.py` |

### Model summary

| Model | Input | Hidden units | Epochs (default) | Output file |
|-------|-------|--------------|------------------|-------------|
| SimpleRNN | Past `High` (scaled) | 125 | 10 | `output/model_rnn.h5` |
| LSTM | Past `High` (scaled) | 125 | 10 | `output/model_lstm.h5` |
| Multivariate LSTM | 6 features × 1 timestep | 125 | 20 | `output/model_mv_lstm.h5` |

## Notebook workflow

Prefer Jupyter? Open `lib/lstm_p1.ipynb` in Jupyter, VS Code, or [Google Colab](https://colab.research.google.com/) after uploading the project files. The notebook covers the same concepts as the modular scripts.

## Customization

Edit constants at the top of `engine.py`:

- `tstart` / `tend` — training date range
- `n_steps` — lookback window length
- `steps_in_future` — autoregressive forecast horizon for RNN/LSTM
- `epochs`, `batch_size` — passed into training functions

Swap in your own CSV with columns `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` and update the file path in `engine.py`.

## Known limitations

- Single ticker (AAPL) and a fixed train/test split.
- Univariate models use `n_steps=1`, so each prediction only sees one prior day.
- Market regimes, news, and macro factors are not modeled.
- Past performance on historical data does not guarantee future accuracy.

## License

Use and modify for educational purposes. Verify dependencies and versions match your environment before deploying anywhere beyond local experiments.
