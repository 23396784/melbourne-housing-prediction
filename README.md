# Melbourne Housing Dataset

## Overview

This directory contains the Melbourne housing dataset used for price prediction modeling.

## Dataset Description

| Attribute | Description |
|-----------|-------------|
| **Records** | 150 properties |
| **Suburbs** | Richmond, South Yarra, Hawthorn |
| **Time Period** | 2001-2025 |
| **Source** | Manually collected from Melbourne real estate sources |

## Features

### Original Features (15)
The raw dataset contained 15 features, which were reduced to 6 key variables after quality assessment.

### Final Features (6)

| Feature | Type | Description |
|---------|------|-------------|
| `Suburb` | Categorical | Property suburb (Richmond, South Yarra, Hawthorn) |
| `Property_Type` | Categorical | House, Apartment, or Townhouse |
| `Bedrooms` | Numeric | Number of bedrooms (1-7) |
| `Bathrooms` | Numeric | Number of bathrooms (1-5) |
| `Car_Spaces` | Numeric | Number of car parking spaces |
| `Distance_to_CBD` | Numeric | Distance to Melbourne CBD in kilometers |

### Target Variable

| Feature | Type | Description |
|---------|------|-------------|
| `Sold_Price` | Numeric | Property sale price in AUD |

## Data Quality

### Removed Features
- `Building_Area` - Excessive missing values
- `Year_Built` - Excessive missing values
- `Days_on_Market` - Excessive missing values
- `Property_ID` - Non-predictive identifier
- `Address` - Redundant location information
- `Agency` - Non-predictive categorical

### Data Cleaning Applied
- Removed duplicate entries
- Handled missing values in critical fields
- Corrected date formatting issues
- Filtered extreme price outliers (1st-99th percentile)
- Balanced suburb representation

## Summary Statistics

### Price Distribution
- **Minimum**: $206,000
- **Maximum**: $509,067
- **Mean**: $330,628
- **Median**: $325,111

### By Suburb

| Suburb | Count | Avg Price | Min Price | Max Price |
|--------|-------|-----------|-----------|-----------|
| Richmond | 39 | $315,577 | $206,000 | $482,913 |
| South Yarra | 37 | $320,561 | $220,173 | $482,445 |
| Hawthorn | 30 | $355,746 | $232,505 | $509,067 |

### By Property Type

| Type | Count | Avg Price |
|------|-------|-----------|
| House | 28 | $615,246 |
| Apartment | 106 | $313,666 |
| Townhouse | 16 | $519,955 |

## Usage

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/melbourne_housing.csv')

# View basic info
print(df.info())
print(df.describe())
```

## Data Privacy

This dataset contains publicly available property sale information. No personally identifiable information is included.

## License

This dataset is provided for educational and research purposes only.
