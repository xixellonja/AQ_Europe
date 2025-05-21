import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


data = pd.read_csv(r'/home/stud/kellezi/data/2018 unvalidated.csv')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Drop unnecessary columns
data = data[['Air Pollution Level', 'Air Quality Station Area', 'Air Quality Station Type', 
             'Country', 'Latitude', 'Longitude', 'Altitude', 'City Population', 'City']]

# filter Europe (latitude and longitude bounds)
europe_lat_min, europe_lat_max = 35, 72
europe_lon_min, europe_lon_max = -25, 60
data = data[
    (data['Latitude'] >= europe_lat_min) & (data['Latitude'] <= europe_lat_max) &
    (data['Longitude'] >= europe_lon_min) & (data['Longitude'] <= europe_lon_max)
]

# Categorize PM2.5 levels
def categorize_pm25(pm25_value):
    if pm25_value <= 5:
        return 'Excellent'
    elif pm25_value <= 10:
        return 'Good'
    elif pm25_value <= 15:
        return 'Moderate'
    elif pm25_value <= 25:
        return 'Unhealthy'
    elif pm25_value <= 35:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

data['PM2.5_Category'] = data['Air Pollution Level'].apply(categorize_pm25)

# Encode categorical variables
pm25_categories = ['Hazardous', 'Very Unhealthy', 'Unhealthy', 'Moderate', 'Good', 'Excellent']
ordinal_encoder = OrdinalEncoder(categories=[pm25_categories])
data['PM2.5_Label'] = ordinal_encoder.fit_transform(data[['PM2.5_Category']])

# One-hot encode 'Air Quality Station Area' and 'Air Quality Station Type'
data_encoded = pd.get_dummies(data, columns=['Air Quality Station Area', 'Air Quality Station Type'], dtype=float)

# Frequency encode 'Country' and 'City'
data_encoded['Country_Freq'] = data['Country'].map(data['Country'].value_counts(normalize=True))
data_encoded['City_Frequency'] = data['City'].map(data['City'].value_counts(normalize=True))

# Drop unnecessary columns and fill missing values
data_encoded.drop(['Country', 'PM2.5_Category', 'Air Pollution Level', 'City'], axis=1, inplace=True)
data_encoded['City Population'] = data_encoded['City Population'].fillna(0)
data_encoded['City_Frequency'] = data_encoded['City_Frequency'].fillna(0)

processed_data = data_encoded.copy()

print('Missing values on raw data:')
print(data.isnull().sum())

print('Country list:')
print(data['Country'].sort_values().unique())


print('Total Cities count:')
print(data['City'].nunique())

print('Ranges of quantitative data:')
print(data.describe())


# Correlation with target var
corr = data_encoded.corr()
print(corr['PM2.5_Label'].sort_values(ascending=False))


# PM2.5_Label distribution
print("PM2.5_Label distribution:")
print(data_encoded['PM2.5_Label'].value_counts())


#drop city attrb
data_encoded.drop(['City Population', 'City_Frequency'], axis=1, inplace=True)


# Fill with 0 all NaN's
#data_encoded['City Population'] = data_encoded['City Population'].fillna(0)
#data_encoded['City_Freq'] = data_encoded['City_Freq'].fillna(0)

# Drop NaNs from city attrb
#data_encoded.dropna(subset=['City Population', 'City_Freq'], inplace=True)

#drop city frequency only
#data_encoded.dropna(subset=['City Population'], inplace=True)
#data_encoded.drop(['City_Freq'], axis=1, inplace=True)

#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 0]
#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 1]
#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 5]


processed_data.to_csv('processed_unval_data.csv', index=False)


# https://docs.python.org/3/library/functions.html
# https://docs.python.org/3/library
# https://matplotlib.org/stable/api/pyplot_api.html
# https://seaborn.pydata.org/api.html
