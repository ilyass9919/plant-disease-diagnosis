plant-disease-diagnosis/
├── training/                    # Kaggle/Colab notebook lives here
│   ├── train_plantdoc.ipynb     # Full training notebook (runs on Kaggle)
│   ├── configs/
│   │   └── model_config.yaml    # Hyperparams, class map, thresholds
│   └── exports/                 # Where you drop the .keras file after training
│
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── predict.py
│   ├── services/
│   │   ├── inference.py         # Model loading + prediction
│   │   ├── uncertainty.py       # Threshold logic
│   │   └── report/
│   │       ├── static_report.py     # Current: hardcoded summaries
│   │       └── agent_report.py      # Future: LLM agent (stub)
│   ├── models/
│   │   └── model_loader.py
│   ├── storage/                 # Continuous improvement data layer
│   │   └── prediction_store.py  # Logs image + output + metadata
│   └── schemas/
│       └── response.py          # Pydantic models for API I/O
│
├── saved_models/
│   └── registry.json            # Model version metadata
├── data/                        # Local copy of filtered dataset (optional)
├── requirements.txt
└── .env