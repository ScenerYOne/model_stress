# Preprocess EDA and PPG Signal Using Python
This project focuses on processing and analyzing EDA (Electrodermal Activity) and PPG (Photoplethysmography) signals using Python, primarily utilizing the NeuroKit2 library for biomedical signal processing and analysis.

## Installation and Setup
```
# Clone the repository
git clone https://github.com/Jamessurapat26/preprocess-eda-ppg.git

# Install required packages
pip install -r requirements.txt

# unzip raw data
unzip Raw.zip
```

## Project Structure

```
preprocess-eda-ppg
├── ppg-process.py       # PPG processing script
├── eda-process.py       # EDA processing script
├── concat-ppg.py        # PPG data concatenation
├── concat-eda.py        # EDA data concatenation
├── label.json           # Subject metadata
├── Raw/                    # Raw signal data
│   ├── ppg/               # Raw PPG files
│   └── eda/               # Raw EDA files
├─ requirements.txt
└─ test.py
```
### Processing Pipeline
1. Process PPG signals:
```bash
python ppg-process.py
```

2. Process EDA signals:
```bash
python eda-process.py
```

3. Combine processed data:
```bash
python concat-ppg.py
python concat-eda.py
```

## File Format Requirements

### PPG Files
- Naming: `s##_PPG.csv` (e.g., s01_PPG.csv)
- Columns:
  - LocalTimestamp: Unix timestamp
  - PG: PPG signal values
  - Quality: Signal quality indicator

### EDA Files
- Naming: `S##_EDA.csv` (e.g., S01_EDA.csv)
- Columns:
  - LocalTimestamp: Unix timestamp
  - EDA: EDA signal values
