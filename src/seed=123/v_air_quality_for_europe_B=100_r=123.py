import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt  
import seaborn as sns 


from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import Badge, TypiClust
from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices

from copkmeans.cop_kmeans import cop_kmeans  


data = pd.read_csv("/home/stud/kellezi/data/Annual mean (hourly) 2018.csv")
#data = pd.read_csv("/home/stud/kellezi/data/2018 unvalidated.csv")

#data = pd.read_csv("/home/stud/kellezi/data/Annual mean (hourly) 2018.csv")
#data = pd.read_csv("/home/stud/kellezi/data/Annual mean (hourly) 2018.csv")
#data = pd.read_csv('/kaggle/input/eea-all-2018/2018 unvalidated.csv')

# Global random variable 
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


########################################  data overview
data.isnull().sum()
data.info()

# Drop unnecessary columns
data = data[['Air Pollution Level', 'Air Quality Station Area', 'Air Quality Station Type', 'Country', 'Latitude', 'Longitude', 'Altitude', 'City Population', 'City']]
print(data.columns)

# Check for duplicates
data.duplicated().sum()


# country and city
data['Country'].sort_values().unique()
data['City'].nunique()

#air polution level range
pm_max = data['Air Pollution Level'].max() 
pm_min = data['Air Pollution Level'].min()

#city population range
cp_max = data['City Population'].max() 
cp_min = data['City Population'].min()

#latitude and longitude bounds for Europe
europe_lat_min, europe_lat_max = 35, 72
europe_lon_min, europe_lon_max = -25, 60

#include EU countries only
data = data[
    (data['Latitude'] >= europe_lat_min) & (data['Latitude'] <= europe_lat_max) &
    (data['Longitude'] >= europe_lon_min) & (data['Longitude'] <= europe_lon_max)]

#Altitude range
alt_max = data['Altitude'].max() 
alt_min = data['Altitude'].min()

print("PM2.5 level: " + str(pm_min) + " to " + str(pm_max))
print("Latitude range: " + str(europe_lat_min) + " to " + str(europe_lat_max))
print("Longitude range: " + str(europe_lon_min) + " to " + str(europe_lon_max))
print("Altitude range: " + str(alt_min) + " to " + str(alt_max))
print("City Population: " + str(cp_min) + " to " + str(cp_max))


data.describe()




#################################### Encoding categorical variables

# Categorie pm25 
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
    elif pm25_value >= 35:
        return 'Hazardous'


data['PM2.5_Category'] = data['Air Pollution Level'].apply(categorize_pm25)
print(data)
data_encoded = data

#(ordinal) PM2.5 levels
pm25_categories = ['Hazardous', 'Very Unhealthy', 'Unhealthy', 'Moderate', 'Good', 'Excellent']
ordinal_encoder = OrdinalEncoder(categories=[pm25_categories])
data_encoded['PM2.5_Label'] = ordinal_encoder.fit_transform(data[['PM2.5_Category']])

#(ordinal) PM2.5 levels
#* pm25_label = {
#    'Excellent': 0,
#    'Good': 1,
#    'Moderate': 2,    
#    'Unhealthy': 3,
#    'Very Unhealthy': 4,
#   'Hazardous': 5
#}
#data['PM2.5_Label'] = data['PM2.5_Category'].map(pm25_label)


# One-Hot Encode 'Air Quality Station Area' and 'Air Quality Station Type'              #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
data_encoded = pd.get_dummies(data, columns=['Air Quality Station Area', 'Air Quality Station Type'], dtype=float)

# Frequency Encode 'Country'
country_freq = data['Country'].value_counts(normalize=True)
data_encoded['Country_Freq'] = data['Country'].map(country_freq)

#Frequency Encode 'City'
country_freq = data['City'].value_counts(normalize=True)
data_encoded['City_Frequency'] = data['City'].map(country_freq)

# Drop unnecessary columns
data_encoded.drop(['Country', 'PM2.5_Category', 'Air Pollution Level', 'City'], axis=1, inplace=True)

# Fill with 0 all NaN's
data_encoded['City Population'] = data_encoded['City Population'].fillna(0)
data_encoded['City_Frequency'] = data_encoded['City_Frequency'].fillna(0)

print(data_encoded)


# Correlation matrix
correlation_matrix = data_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.savefig('correlation_matrix_validated.png', bbox_inches='tight')
plt.show()

# PM2.5_Label distribution
print("PM2.5_Label distribution:")
print(data_encoded['PM2.5_Label'].value_counts())


#### CITY AND CITY POPULATION
# Fill with 0 all NaN's
#data_encoded['City Population'] = data_encoded['City Population'].fillna(0)
#data_encoded['City_Freq'] = data_encoded['City_Freq'].fillna(0)

# Drop NaNs from city attrb
#data_encoded.dropna(subset=['City Population', 'City_Freq'], inplace=True)

#drop city frequency only
#data_encoded.dropna(subset=['City Population'], inplace=True)
#data_encoded.drop(['City_Freq'], axis=1, inplace=True)

#drop city attrb
data_encoded.drop(['City Population', 'City_Frequency'], axis=1, inplace=True)


#### TEST WITH LESS LABELS
#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 0]
#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 1]
#data_encoded = data_encoded[data_encoded['PM2.5_Label'] != 5]



total_entries = len(data_encoded)

urban_pm = (data_encoded.loc[data_encoded["Air Quality Station Area_Urban"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100
suburban_pm = (data_encoded.loc[data_encoded["Air Quality Station Area_Suburban"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100
rural_pm = (data_encoded.loc[data_encoded["Air Quality Station Area_Rural"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100

background_pm = (data_encoded.loc[data_encoded["Air Quality Station Type_Background"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100
traffic_pm = (data_encoded.loc[data_encoded["Air Quality Station Type_Traffic"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100
industrial_pm = (data_encoded.loc[data_encoded["Air Quality Station Type_Industrial"] == 1, 'PM2.5_Label'].value_counts() / total_entries) * 100

print("Urban PM2.5:\n", urban_pm.sort_values(), len(urban_pm)/len(data_encoded))
print("Suburban PM2.5:\n", suburban_pm.sort_values())
print("Rural PM2.5:\n", rural_pm.sort_values())
print("Background PM2.5:\n", background_pm.sort_values())
print("Traffic PM2.5:\n", traffic_pm.sort_values())
print("Industrial PM2.5:\n", industrial_pm.sort_values())

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# https://machinelearninggeek.com/multi-layer-perceptron-neural-network-using-python/
# https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.LearningCurveDisplay.html#sklearn.model_selection.LearningCurveDisplay.from_estimator
# https://scikit-learn.org/dev/modules/learning_curve.html#learning-curve
# https://scikit-learn.org/dev/auto_examples/model_selection/plot_learning_curve.html
# https://scikit-learn.org/dev/modules/model_evaluation.html#scoring-parameter
# https://scikit-learn.org/1.5/modules/generated/sklearn.utils.shuffle.html
# https://www.youtube.com/watch?v=2Bkp4B8sJ2Y

# Features and target variable
X = data_encoded.drop('PM2.5_Label', axis=1)
y = data_encoded['PM2.5_Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Normalize features
#scaler = MinMaxScaler()
#X_normalized = scaler.fit_transform(X)
#X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

# Split the data - 20% for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    #X_normalized,
    y,
    test_size = 0.2,
    random_state = RANDOM_STATE,
    #stratify = y # makes sure the same class label distribution is same as in the OG dataset. check how and why it effects learning curve -> less var
)

# reset indices after splitting
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# shuffle again for more randomness (:
X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=RANDOM_STATE)

print('Training data consists of ' + str(len(X_train_shuffled.columns.unique())) + ' features')




######################## Define MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes = (3,),  
    max_iter = 4000,
    random_state = RANDOM_STATE,
    #alpha = 0.1
    #solver = "lbfgs", # smaller var in training and valid. score - best for small datasets
)


# Fit the model on full training data
mlp.fit(X_train_shuffled, y_train_shuffled)
y_pred_full = mlp.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)
print(f"\nFull dataset training accuracy: {accuracy_full:.4f}")


display = LearningCurveDisplay.from_estimator(
    estimator = mlp,
    X = X_train_shuffled,
    y = y_train_shuffled,
    train_sizes = np.linspace(0.1, 1.0, 5), # also by default
    #train_sizes = [10, 50, 100, 150, 200, 250, 322],
    #cv = 2, # 5 by default, but class label 5 has only 3 instances 
    cv = StratifiedKFold(n_splits=2),
    score_type = 'both',
    #n_jobs = -1, # full pwr
    shuffle = True, # shuffles training data makes sure each cross-validation fold is representative of the overall pm2.5 label distribution
    random_state=RANDOM_STATE,
)

#Plt learning curve display 
#plt.figure(figsize=(8, 6))



plt.title('MLP Classifier Performance with K-Fold Cross-Validation')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.axhline(y=accuracy_full, color='r', linestyle='--', label='Full Dataset')
plt.grid(True)
plt.legend()
plt.savefig('valscore_random123_.png', bbox_inches='tight')
plt.show()





























#################################################### BATCH SIZE 100
######################### BASELINE MODEL


batch_size = 100 
batch_nr = 7 

accuracies_baseline = []
train_sizes_baseline = []

for batch in range(1, batch_nr + 1):
    current_training_size = batch * batch_size
    if current_training_size > len(X_train_shuffled):
        current_training_size = len(X_train_shuffled)
        
    X_train_baseline = X_train_shuffled[:current_training_size]
    y_train_baseline = y_train_shuffled[:current_training_size]
    train_sizes_baseline.append(current_training_size)

    mlp.fit(X_train_baseline, y_train_baseline)

    y_pred_base = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred_base)
    accuracies_baseline.append(acc)
    
    print(f"Accurancy score: {acc} - Training size: {current_training_size}")


label_accuracies_baseline = {}
for label in np.unique(y_test):
    label_indices = y_test == label
    label_acc = accuracy_score(y_test[label_indices], y_pred_base[label_indices])
    label_accuracies_baseline[label] = label_acc



plt.figure(figsize=(8, 6))
plt.plot(train_sizes_baseline, accuracies_baseline, marker='o', color='orange', label="Random")
plt.axhline(y=accuracy_full, color='red', linestyle='--', label='Full Dataset')
plt.xlabel('Number of Training Samples', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('MLP Classifier Performance (qs = Random)', fontsize=14)
plt.grid(True)
plt.legend()
#plt.savefig('Baseline Accuracy Score_random123.png', bbox_inches='tight')
plt.show()


##################################### AL BADGE

# AL parameters
M = 100  
X_target = len(X)  
qs = Badge()
B = 100  


# df to array for AL
X = X_train_shuffled.values  
y_true_badge = y_train_shuffled.values  

#  MISSING_LABEL for np.nan on target var
MISSING_LABEL = np.nan 
y = np.full(y_true_badge.shape, MISSING_LABEL, dtype=float)

# assign true labels to M intial points
initial = np.random.choice(len(X), size=M, replace=False)
y[initial] = y_true_badge[initial]

# MLPClassifier for AL
badge_mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=RANDOM_STATE)
badge_clf = SklearnClassifier(estimator=badge_mlp, classes=np.unique(y_true_badge), missing_label=MISSING_LABEL)


iteration_acc = []
labeled_samples = []
iteration = 0

while len(labeled_indices(y)) < X_target:
    iteration += 1
    print(f"AL Iteration {iteration}")

    # fit on labeled data
    labeled_idx = labeled_indices(y)
    badge_clf.fit(X[labeled_idx], y[labeled_idx])

    
    # save accuracy
    y_pred_badge = badge_clf.predict(X)
    acc = accuracy_score(y_true_badge, y_pred_badge)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_idx))
    print(f"Accuracy: {acc:.2f}, Labeled Samples: {len(labeled_idx)}")


    # query next batch
    unlabeled_idx = unlabeled_indices(y)
    if len(unlabeled_idx) == 0:
        print("All data points have been labeled.")
        break

    query_idx = qs.query(X=X, y=y, clf=badge_clf, candidates=unlabeled_idx, batch_size=B)

    # map query idx to OG dataset and assign true labels
    query = query_idx
    y[query] = y_true_badge[query]

# record last batch
if len(labeled_indices(y)) == X_target:
    y_pred_badge = badge_clf.predict(X)
    acc = accuracy_score(y_true_badge, y_pred_badge)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_indices(y)))
    print(f"Final Accuracy: {acc:.2f}, Total Labeled Samples: {len(labeled_indices(y))}")

badge_x_100, badge_y_100 = labeled_samples, iteration_acc
acc_badge = iteration_acc[-1]
 
label_accuracies_badge = {}
for label in np.unique(y_true_badge):
    label_indices = y_true_badge == label
    label_acc = accuracy_score(y_true_badge[label_indices], y_pred_badge[label_indices])
    label_accuracies_badge[label] = label_acc


plt.figure(figsize=(8, 6))
#plt.plot(badge_x_200, badge_y_200, marker='o', color='grey',  label='B=200')
plt.plot(badge_x_100, badge_y_100, marker='o', color='green',  label='qs=Badge')
#plt.plot(badge_x_50, badge_y_50, marker='o', color='purple',  label='B=50')
#plt.plot(badge_x_20, badge_y_20, marker='o', color='blue',  label='B=20')
plt.plot(train_sizes_baseline, accuracies_baseline, color='orange', linestyle='--',   label="qs=random")
plt.title('MLP Classifier Performance (B=100)(seed=123)', fontsize=12)
plt.xlabel("Number of Labeled Instances", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid()
plt.legend()
#plt.savefig('123AL Learning Accuracy Score__Badge_100_123.png', bbox_inches='tight')
plt.show()




#################################################### CC * TYPICLUST

#https://stackoverflow.com/questions/73294520/clustering-algorithm-that-keeps-separated-a-given-set-of-pairs


M = 100

B_tc = 20  # Batch size for TypiClust
cop_clusters  = 5

cannot_link = [] 
must_link= [] 
k=1

# df to array for AL
X = X_train_shuffled.values  
y_true_tc5 = y_train_shuffled.values 


# indices for AQ Area and Type
urban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Urban"] == 1.0].index
suburban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Suburban"] == 1.0].index
rural = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Rural"] == 1.0].index
traffic = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Traffic"] == 1.0].index
background = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Background"] == 1.0].index
industrial = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Industrial"] == 1.0].index


# cannot link: urban+traffic vs rural+background indices
urban_traffic = np.intersect1d(urban, traffic)
rural_background = np.intersect1d(rural, background)

for i in urban_traffic:
    for j in rural_background:
        cannot_link.append((i, j))


# must link for AQ Area and Type
for indices in [background, industrial, traffic]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))

for indices in [urban, suburban, rural]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))


# create clusters (cluster_labels: contans idx and cc labels, cluster_centers: the cluster k coord)
cluster_labels, cluster_centers = cop_kmeans(X, k=cop_clusters , ml=must_link, cl=cannot_link)

#create a dict with cluster label and list of resp. indices for AL
cluster_dict = {}
unique_cluster_labels = np.unique(cluster_labels)
for label in unique_cluster_labels:
    indices = np.where(cluster_labels == label)[0]
    cluster_dict[label] = indices



# target var is set to missing label('unlabeled')
MISSING_LABEL = np.nan
y = np.full(y_true_tc5.shape, MISSING_LABEL, dtype=float)


# assign true labels to M intial points
initial = np.random.choice(len(X), size=M, replace=False)
y[initial] = y_true_tc5[initial]


# MLPClassifier for AL
tc5_mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=RANDOM_STATE)
tc5_clf = SklearnClassifier(estimator=tc5_mlp, classes=np.unique(y_true_tc5), missing_label=MISSING_LABEL)

# query strategy
qs = TypiClust(k=k)

# AL Loop
iteration_acc_tc5 = []
labeled_samples = []
iteration = 0

while len(labeled_indices(y)) < X_target:
    iteration += 1
    print(f"AL CC Iteration {iteration}")

   # fit on labeled data
    labeled_idx = labeled_indices(y)
    tc5_clf.fit(X[labeled_idx], y[labeled_idx])

    # save accuracy
    y_pred_tc5 = tc5_clf.predict(X)
    acc = accuracy_score(y_true_tc5, y_pred_tc5)
    iteration_acc_tc5.append(acc)
    labeled_samples.append(len(labeled_idx))
    print(f"Accuracy: {acc:.2f}, Labeled Samples: {len(labeled_idx)}")


    # Query the next batch using TypiClust from different clusters
    unlabeled_idx = unlabeled_indices(y)
    if len(unlabeled_idx) == 0:
        print("All data points have been labeled.")
        break

    # Select B unlabeled instance via typiclust from each cluster
    query_idx = []
    for cluster_id, instances in cluster_dict.items():
        unlabeled_instances = np.intersect1d(instances, unlabeled_idx)
        if unlabeled_instances.size > 0:
            cluster_query = qs.query(X=X[unlabeled_instances], y=y[unlabeled_instances], batch_size=B_tc)
            query_idx.append(unlabeled_instances[cluster_query])

    # make sure query batch size is not exceeded 
    #query_idx = query_idx[:B]

    # label the query 
    query_idx = np.concatenate(query_idx) #sequence to 1d array
    y[query_idx] = y_true_tc5[query_idx]


# record last batch
if len(labeled_indices(y)) == X_target:
    y_pred_tc5 = tc5_clf.predict(X)
    acc = accuracy_score(y_true_tc5, y_pred_tc5)
    iteration_acc_tc5.append(acc)
    labeled_samples.append(len(labeled_indices(y)))
    print(f"Final Accuracy: {acc:.2f}, Total Labeled Samples: {len(labeled_indices(y))}")

tc_x_c5_20, tc_y_c5_20 = labeled_samples, iteration_acc_tc5
acc_tc5 = iteration_acc_tc5[-1]



label_accuracies_tc_5 = {}
for label in np.unique(y_true_tc5):
    label_indices = y_true_tc5 == label
    label_acc = accuracy_score(y_true_tc5[label_indices], y_pred_tc5[label_indices])
    label_accuracies_tc_5[label] = label_acc


colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster_id, indices in cluster_dict.items():
    ax.scatter(
        data_encoded.iloc[indices]['Longitude'], 
        data_encoded.iloc[indices]['Latitude'], 
        data_encoded.iloc[indices]['Altitude'], 
        label=f'Cluster {cluster_id}', 
        alpha=0.7, 
        color=colors[cluster_id]
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude")
ax.set_title("Constrained Clusters (cop clusters=5)")
ax.legend()
plt.savefig("123_constrained_clusters_c5.png", bbox_inches="tight")
plt.show()

###############
M = 100

B_tc = 33  # Batch size for TypiClust
cop_clusters  = 3

cannot_link = [] 
must_link= [] 


# df to array for AL
X = X_train_shuffled.values  
y_true_tc3 = y_train_shuffled.values 


# indices for AQ Area and Type
urban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Urban"] == 1.0].index
suburban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Suburban"] == 1.0].index
rural = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Rural"] == 1.0].index
traffic = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Traffic"] == 1.0].index
background = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Background"] == 1.0].index
industrial = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Industrial"] == 1.0].index


# cannot link: urban+traffic vs rural+background indices
urban_traffic = np.intersect1d(urban, traffic)
rural_background = np.intersect1d(rural, background)

for i in urban_traffic:
    for j in rural_background:
        cannot_link.append((i, j))


# must link for AQ Area and Type
for indices in [background, industrial, traffic]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))

for indices in [urban, suburban, rural]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))


# create clusters (cluster_labels: contans idx and cc labels, cluster_centers: the cluster k coord)
cluster_labels, cluster_centers = cop_kmeans(X, k=cop_clusters , ml=must_link, cl=cannot_link)

#create a dict with cluster label and list of resp. indices for AL
cluster_dict = {}
unique_cluster_labels = np.unique(cluster_labels)
for label in unique_cluster_labels:
    indices = np.where(cluster_labels == label)[0]
    cluster_dict[label] = indices



# target var is set to missing label('unlabeled')
MISSING_LABEL = np.nan
y = np.full(y_true_tc3.shape, MISSING_LABEL, dtype=float)


# assign true labels to M intial points
initial = np.random.choice(len(X), size=M, replace=False)
y[initial] = y_true_tc3[initial]


# MLPClassifier for AL
tc3_mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=RANDOM_STATE)
tc3_clf = SklearnClassifier(estimator=tc3_mlp, classes=np.unique(y_true_tc3), missing_label=MISSING_LABEL)
# query strategy
qs = TypiClust(k=k)

# AL Loop
iteration_acc = []
labeled_samples = []
iteration = 0

while len(labeled_indices(y)) < X_target:
    iteration += 1
    print(f"AL CC Iteration {iteration}")

   # fit on labeled data
    labeled_idx = labeled_indices(y)
    tc3_clf.fit(X[labeled_idx], y[labeled_idx])

    # save accuracy
    y_pred_tc3 = tc3_clf.predict(X)
    acc = accuracy_score(y_true_tc3, y_pred_tc3)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_idx))
    print(f"Accuracy: {acc:.2f}, Labeled Samples: {len(labeled_idx)}")


    # Query the next batch using TypiClust from different clusters
    unlabeled_idx = unlabeled_indices(y)
    if len(unlabeled_idx) == 0:
        print("All data points have been labeled.")
        break

    # Select B unlabeled instance via typiclust from each cluster
    query_idx = []
    for cluster_id, instances in cluster_dict.items():
        unlabeled_instances = np.intersect1d(instances, unlabeled_idx)
        if unlabeled_instances.size > 0:
            cluster_query = qs.query(X=X[unlabeled_instances], y=y[unlabeled_instances], batch_size=B_tc)
            query_idx.append(unlabeled_instances[cluster_query])

    # make sure query batch size is not exceeded 
    #query_idx = query_idx[:B]

    # label the query 
    query_idx = np.concatenate(query_idx) #sequence to 1d array
    y[query_idx] = y_true_tc3[query_idx]

# record last batch
if len(labeled_indices(y)) == X_target:
    y_pred_tc3 = tc3_clf.predict(X)
    acc = accuracy_score(y_true_tc3, y_pred_tc3)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_indices(y)))
    print(f"Final Accuracy: {acc:.2f}, Total Labeled Samples: {len(labeled_indices(y))}")

tc_x_c3_33, tc_y_c3_33 = labeled_samples, iteration_acc
acc_tc3 = iteration_acc[-1]

#record label accuracies
label_accuracies_tc_3 = {}
for label in np.unique(y_true_tc3):
    label_indices = y_true_tc3 == label
    label_acc = accuracy_score(y_true_tc3[label_indices], y_pred_tc3[label_indices])
    label_accuracies_tc_3[label] = label_acc


colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster_id, indices in cluster_dict.items():
    ax.scatter(
        data_encoded.iloc[indices]['Longitude'], 
        data_encoded.iloc[indices]['Latitude'], 
        data_encoded.iloc[indices]['Altitude'], 
        label=f'Cluster {cluster_id}', 
        alpha=0.7, 
        color=colors[cluster_id]
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude")
ax.set_title("Constrained Clusters (cop clusters=3)")
ax.legend()
plt.savefig("123_constrained_clusters_c3.png", bbox_inches="tight")
plt.show()


############################################################################
M = 100

B_tc = 14  # Batch size for TypiClust
cop_clusters  = 7

cannot_link = [] 
must_link= [] 


# df to array for AL
X = X_train_shuffled.values  
y_true_tc7 = y_train_shuffled.values 


# indices for AQ Area and Type
urban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Urban"] == 1.0].index
suburban = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Suburban"] == 1.0].index
rural = X_train_shuffled[X_train_shuffled["Air Quality Station Area_Rural"] == 1.0].index
traffic = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Traffic"] == 1.0].index
background = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Background"] == 1.0].index
industrial = X_train_shuffled[X_train_shuffled["Air Quality Station Type_Industrial"] == 1.0].index


# cannot link: urban+traffic vs rural+background indices
urban_traffic = np.intersect1d(urban, traffic)
rural_background = np.intersect1d(rural, background)

for i in urban_traffic:
    for j in rural_background:
        cannot_link.append((i, j))


# must link for AQ Area and Type
for indices in [background, industrial, traffic]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))

for indices in [urban, suburban, rural]:
    if len(indices) > 1:
        for i in range(len(indices) - 1):
            must_link.append((indices[i], indices[i + 1]))


# create clusters (cluster_labels: contans idx and cc labels, cluster_centers: the cluster k coord)
cluster_labels, cluster_centers = cop_kmeans(X, k=cop_clusters , ml=must_link, cl=cannot_link)

#create a dict with cluster label and list of resp. indices for AL
cluster_dict = {}
unique_cluster_labels = np.unique(cluster_labels)
for label in unique_cluster_labels:
    indices = np.where(cluster_labels == label)[0]
    cluster_dict[label] = indices



# target var is set to missing label('unlabeled')
MISSING_LABEL = np.nan
y = np.full(y_true_tc7.shape, MISSING_LABEL, dtype=float)


# assign true labels to M intial points
initial = np.random.choice(len(X), size=M, replace=False)
y[initial] = y_true_tc7[initial]


# MLPClassifier for AL
tc7_mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=RANDOM_STATE)
tc7_clf = SklearnClassifier(estimator=tc7_mlp, classes=np.unique(y_true_tc7), missing_label=MISSING_LABEL)
# query strategy
qs = TypiClust(k=k)

# AL Loop
iteration_acc = []
labeled_samples = []
iteration = 0

while len(labeled_indices(y)) < X_target:
    iteration += 1
    print(f"AL CC Iteration {iteration}")

   # fit on labeled data
    labeled_idx = labeled_indices(y)
    tc7_clf.fit(X[labeled_idx], y[labeled_idx])

    # save accuracy
    y_pred_tc7 = tc7_clf.predict(X)
    acc = accuracy_score(y_true_tc7, y_pred_tc7)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_idx))
    print(f"Accuracy: {acc:.2f}, Labeled Samples: {len(labeled_idx)}")


    # Query the next batch using TypiClust from different clusters
    unlabeled_idx = unlabeled_indices(y)
    if len(unlabeled_idx) == 0:
        print("All data points have been labeled.")
        break

    # Select B unlabeled instance via typiclust from each cluster
    query_idx = []
    for cluster_id, instances in cluster_dict.items():
        unlabeled_instances = np.intersect1d(instances, unlabeled_idx)
        if unlabeled_instances.size > 0:
            cluster_query = qs.query(X=X[unlabeled_instances], y=y[unlabeled_instances], batch_size=B_tc)
            query_idx.append(unlabeled_instances[cluster_query])

    # make sure query batch size is not exceeded 
    #query_idx = query_idx[:B]

    # label the query 
    query_idx = np.concatenate(query_idx) #sequence to 1d array
    y[query_idx] = y_true_tc7[query_idx]

# record last batch
if len(labeled_indices(y)) == X_target:
    y_pred_tc7 = tc7_clf.predict(X)
    acc = accuracy_score(y_true_tc7, y_pred_tc7)
    iteration_acc.append(acc)
    labeled_samples.append(len(labeled_indices(y)))
    print(f"Final Accuracy: {acc:.2f}, Total Labeled Samples: {len(labeled_indices(y))}")

tc_x_c7_14, tc_y_c7_14 = labeled_samples, iteration_acc
acc_tc7 = iteration_acc[-1]

#record label accuracies

label_accuracies_tc_7 = {}
for label in np.unique(y_true_tc7):
    label_indices = y_true_tc7 == label
    label_acc = accuracy_score(y_true_tc7[label_indices], y_pred_tc7[label_indices])
    label_accuracies_tc_7[label] = label_acc


colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred', 'blue']
plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster_id, indices in cluster_dict.items():
    ax.scatter(
        data_encoded.iloc[indices]['Longitude'], 
        data_encoded.iloc[indices]['Latitude'], 
        data_encoded.iloc[indices]['Altitude'], 
        label=f'Cluster {cluster_id}', 
        alpha=0.7, 
        color=colors[cluster_id]
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude")
ax.set_title("Constrained Clusters (cop clusters=7)")
ax.legend()
plt.savefig("123_constrained_clusters_c7.png", bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(badge_x_100, badge_y_100, marker='o', color='green',  label='qs=Badge(B=100)')
plt.plot(tc_x_c7_14, tc_y_c7_14, marker='o',  color= '#8470ff', linestyle='-', label='qs=TypiClust(COP=7;B=14)')
plt.plot(tc_x_c5_20, tc_y_c5_20, marker='o',  color= '#A020F0', linestyle='-', label='qs=TypiClust(COP=5;B=20)')
plt.plot(tc_x_c3_33, tc_y_c3_33, marker='o',  color= '#191970', linestyle='-', label='qs=TypiClust(COP=3;B=33)')
plt.plot(train_sizes_baseline, accuracies_baseline, color='orange', linestyle='--',   label="qs=random")
#plt.title('MLP Classifier Performance with AL and CC (qs = TypiClust)', fontsize=12)
plt.title('MLP Classifier Performance (query_size=100)(seed=123)', fontsize=12)
plt.xlabel("Number of Labeled Instances", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid()
plt.legend()
plt.savefig('ALL_100_123_.png', bbox_inches='tight')
plt.show()







########################################## PAPER PLOTS

# prepare unvalidated dataframe and limits
geo_df = data
data_all = pd.read_csv("/home/stud/kellezi/data/2018 unvalidated.csv")
data_all['PM2.5_Category'] = data_all['Air Pollution Level'].apply(categorize_pm25)

# air polution level range
pm_max_all = data_all['Air Pollution Level'].max() 
pm_min_all = data_all['Air Pollution Level'].min()

# latitude and longitude bounds for Europe
europe_lat_min, europe_lat_max = 35, 72
europe_lon_min, europe_lon_max = -25, 60


data_all = data_all[
    (data_all['Latitude'] >= europe_lat_min) & (data_all['Latitude'] <= europe_lat_max) &
    (data_all['Longitude'] >= europe_lon_min) & (data_all['Longitude'] <= europe_lon_max)]


alt_max_all = data_all['Altitude'].max() 
alt_min_all = data_all['Altitude'].min()



##############  PM2.5 Label 3D plot (EEA validated + EEA unvalidated)


# categories to colors dict
categories = ['Excellent', 'Good', 'Moderate', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
color_map = dict(zip(categories, colors))

data_all['Color'] = data_all['PM2.5_Category'].map(color_map)
geo_df['Color'] = geo_df['PM2.5_Category'].map(color_map)

fig = plt.figure(figsize=(24, 10))

# first subplot for unvalidated data
ax1 = fig.add_subplot(121, projection='3d')
for category in categories:
    idx = geo_df['PM2.5_Category'] == category
    category_geo_df = geo_df.loc[idx]
    ax1.scatter(
        category_geo_df['Longitude'],
        category_geo_df['Latitude'],
        category_geo_df['Altitude'],
        color=color_map[category],
        s=50,
        alpha=0.7
    )

ax1.set_xlim(europe_lon_min, 50)
ax1.set_ylim(europe_lat_min, europe_lat_max)
ax1.set_zlim(alt_min_all, 1200)

ax1.set_xlabel('Longitude', fontsize=14)
ax1.set_ylabel('Latitude', fontsize=14)
ax1.set_zlabel('Altitude', fontsize=14)
ax1.set_title('E1a validated data', fontsize=20)

# second subplot for EEA validated data
ax2 = fig.add_subplot(122, projection='3d')
for category in categories:
    idx = data_all['PM2.5_Category'] == category
    category_data_all = data_all.loc[idx]
    ax2.scatter(
        category_data_all['Longitude'],
        category_data_all['Latitude'],
        category_data_all['Altitude'],
        color=color_map[category],
        s=50,
        alpha=0.7
    )

ax2.set_xlim(europe_lon_min, 50)
ax2.set_ylim(europe_lat_min, europe_lat_max)
ax2.set_zlim(alt_min_all, 1200)

ax2.set_xlabel('Longitude', fontsize=14)
ax2.set_ylabel('Latitude', fontsize=14)
ax2.set_zlabel('Altitude', fontsize=14)
ax2.set_title('ALL data', fontsize=20)


fig.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[category], markersize=10, label=category) for category in categories],
    title='PM2.5 Categories',
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=6,
    prop={'size': 18} 
)

plt.savefig('pm25.png', bbox_inches='tight')
plt.tight_layout()
plt.show()



################################### Air Quality Station
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x="Air Quality Station Area", data=data, hue="Air Quality Station Area", palette="Blues", legend=False)
#sns.countplot(x="Air Quality Station Area", data=data, palette="Blues")
plt.title("Distribution of Air Quality Station Area")
plt.xlabel("Station Area")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.countplot(x="Air Quality Station Type", data=data, hue="Air Quality Station Type", palette="Oranges", legend=False)
#sns.countplot(x="Air Quality Station Type", data=data, palette="Oranges")
plt.title("Distribution of Air Quality Station Type")
plt.xlabel("Station Type")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig('Air Quality Station.png', bbox_inches='tight')
plt.show()

####################################  WHO DATA 2024

# Load the Excel file
wdata = pd.read_excel("/home/stud/kellezi/data/who_ambient_air_quality_database_version_2024_(v6.1).xlsx")

#column_names = wdata.columns.tolist()
#print(column_names)

wdata['PM2.5_Category'] = wdata['pm25_concentration'].apply(categorize_pm25)

wdata['Color'] = wdata['PM2.5_Category'].map(color_map)

vmin = 0  
vmax = 50  

plt.figure(figsize=(12, 8))
plt.scatter(wdata['longitude'], wdata['latitude'], c=wdata['pm25_concentration'], cmap='Reds', alpha=0.7, vmin=vmin, vmax=vmax)
plt.colorbar(label='PM2.5 Level (µg/m³)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of PM2.5 Levels')
plt.savefig('who_pm25.png', bbox_inches='tight')
plt.show()







######### save label accuracies

def round_acc(acc_dict):
    accuracies = {}
    for label, acc in acc_dict.items():
        accuracies[label] = np.round(acc, decimals=2)
    return accuracies

label_accuracies_baseline = np.array(list(round_acc(label_accuracies_baseline).values()))
label_accuracies_badge = np.array(list(round_acc(label_accuracies_badge).values()))
label_accuracies_tc_7 = np.array(list(round_acc(label_accuracies_tc_7).values()))
label_accuracies_tc_5 = np.array(list(round_acc(label_accuracies_tc_5).values()))
label_accuracies_tc_3 = np.array(list(round_acc(label_accuracies_tc_3).values()))

#accuracy_full = np.array(list(round_acc(accuracy_full).values()))
#acc_badge = np.array(list(round_acc(acc_badge).values()))
#label_accuracies_tc_7 = np.array(list(round_acc(label_accuracies_tc_7).values()))
#label_accuracies_tc_5 = np.array(list(round_acc(label_accuracies_tc_5).values()))
#label_accuracies_tc_3 = np.array(list(round_acc(label_accuracies_tc_3).values()))

with open("label_acc_123.txt", "a") as file:
    file.write("Baseline_123: " + str(label_accuracies_baseline) + "\n")
    file.write("Badge_123: " + str(label_accuracies_badge) + "\n")
    file.write("TC7_123: " + str(label_accuracies_tc_7) + "\n")
    file.write("TC5_123: " + str(label_accuracies_tc_5) + "\n")
    file.write("TC3_123: " + str(label_accuracies_tc_3) + "\n")

    file.write("FULL_BL_123: " + str(accuracy_full) + "\n")
    file.write("FULL_BADGE_123: " + str(acc_badge) + "\n")
    file.write("FULL_TC7_123: " + str(acc_tc7) + "\n")
    file.write("FULL_TC5_123: " + str(acc_tc5) + "\n")
    file.write("FULL_TC3_123: " + str(acc_tc3) + "\n")

