## Developed By: Prasannalakshmi G
## Reg No: 212222240075
## Date:


# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING
 


## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

## ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
    
## PROGRAM:
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the Goodreads dataset
data = pd.read_csv('/mnt/data/Goodreads_books(1).csv')

# Convert 'publication_date' to datetime format (if not already) and set it as the index
data['publication_date'] = pd.to_datetime(data['publication_date'], errors='coerce')
data.dropna(subset=['publication_date'], inplace=True)  # Drop rows with invalid dates
data.set_index('publication_date', inplace=True)

# Resample by year to get total 'ratings_count' per year
ratings_data = data[['ratings_count']].resample('Y').sum()

# Display the shape and the first 5 rows of the dataset
print("Shape of the dataset:", ratings_data.shape)
print("First 5 rows of the dataset:")
print(ratings_data.head())

# Plot Original Dataset (Yearly Ratings Count)
plt.figure(figsize=(12, 6))
plt.plot(ratings_data['ratings_count'], label='Yearly Ratings Count', color='blue')
plt.title('Original Goodreads Ratings Data')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.legend()
plt.grid()
plt.show()

# Perform rolling average transformation with a window size of 3 years
rolling_mean_3 = ratings_data['ratings_count'].rolling(window=3).mean()

# Plot Moving Average
plt.figure(figsize=(12, 6))
plt.plot(ratings_data['ratings_count'], label='Original Yearly Ratings Count', color='blue')
plt.plot(rolling_mean_3, label='3-Year Moving Average', color='orange')
plt.title('3-Year Moving Average of Ratings Count')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.legend()
plt.grid()
plt.show()

# Apply Exponential Smoothing (Trend: Additive, No Seasonal Component)
model = ExponentialSmoothing(ratings_data['ratings_count'], trend='add', seasonal=None)
model_fit = model.fit()

# Make predictions for the next 5 years
predictions = model_fit.predict(start=len(ratings_data), end=len(ratings_data) + 4)

# Plot the original data and Exponential Smoothing predictions
plt.figure(figsize=(12, 6))
plt.plot(ratings_data['ratings_count'], label='Original Yearly Ratings Count', color='blue')
plt.plot(predictions, label='Exponential Smoothing Forecast', color='orange')
plt.title('Exponential Smoothing Predictions for Ratings Count')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


```

## OUTPUT:

## Original Plot Data:

![{1811C7D3-A952-4AC0-9620-677303261FC4}](https://github.com/user-attachments/assets/b3d9766f-aacf-4f4f-b9af-568411efbc8d)


## Moving Average:

![{57D9A948-5705-4D1E-B775-FD8E8052B345}](https://github.com/user-attachments/assets/0c8857ca-9366-4860-a67b-021f65883e01)


## Exponential Smoothing:

![{E030F87C-8FEE-410C-9F60-ACC6728ACB20}](https://github.com/user-attachments/assets/bb8c277d-5e0b-46bd-83f7-5fe3047dc62f)



### RESULT:
Thus, The Moving Average Model and Exponential smoothing using python is successfully implemented.
