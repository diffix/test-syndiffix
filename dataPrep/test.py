import pandas as pd
import bz2
import pickle
from syndiffix import Synthesizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})

# Save DataFrame as a bz2 compressed pickle file
with bz2.BZ2File('my_data.pbz2', 'w') as f:
    pickle.dump(df, f)

import requests

def download_and_load(url):
    response = requests.get(url)
    data = bz2.decompress(response.content)
    df = pickle.loads(data)
    return df

target = 'duration'
df = download_and_load('http://open-diffix.org/datasets/loan.pbz2')
# Let's drop 'loan_id' because we know it is of no predictive value
df = df.drop(columns=["loan_id"])
# Change date to a float because DecisionTreeClassifier requires it
df['date'] = df['date'].astype('int64') / 10**9
# Make the PID dataframe
df_pid = df[['account_id']]
# Drop the PID from the dataset because it also has no predictive value
df = df.drop(columns=["account_id"])

# Build the synthesized data
df_syn_no = Synthesizer(df, pids=df_pid).sample()
df_syn = Synthesizer(df, pids=df_pid, target_column=target).sample()

# Split the DataFrame into features (X) and the target variable (y)
X = df.drop(target, axis=1)
y = df[target]
X_syn = df_syn.drop(target, axis=1)
y_syn = df_syn[target]
X_syn_no = df_syn_no.drop(target, axis=1)
y_syn_no = df_syn_no[target]

# And we need to convert strings to one-hot encoding
X = pd.get_dummies(X)
print("Original:")
print(X.head())
X_syn = pd.get_dummies(X_syn)
print("Synthetic Target:")
print(X_syn.head())
X_syn_no = pd.get_dummies(X_syn_no)
print("Synthetic No Target:")
print(X_syn_no.head())

# Split the original dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Do the same for the synthetic data, but noting that we'll use the original test set for testing both original and synthetic
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(X_syn, y_syn, test_size=0.3, random_state=42)

# Do the same for the synthetic data, but noting that we'll use the original test set for testing both original and synthetic
X_train_syn_no, X_test_syn_no, y_train_syn_no, y_test_syn_no = train_test_split(X_syn_no, y_syn_no, test_size=0.3, random_state=42)

def runModel(X_train, X_test, y_train, y_test, dataSource):
    # Create a decision tree classifier and fit it to the training data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Use the trained classifier to make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy {dataSource} is {accuracy}")

runModel(X_train, X_test, y_train, y_test, "Original")
runModel(X_train_syn, X_test, y_train_syn, y_test, "Synthetic Target")
# This last doesn't work because we had generated new symbols when we
# synthesized
#runModel(X_train_syn_no, X_test, y_train_syn_no, y_test, "Synthetic No Target")

quit()

print(f'The accuracy of the model is {accuracy:.2f}')
import scipy.stats
print("orig amount/duration", scipy.stats.spearmanr(df['amount'], df['duration']))
print("amount/payments", scipy.stats.spearmanr(df['amount'], df['payments']))
print("amount/loan_id", scipy.stats.spearmanr(df['amount'], df['loan_id']))

dfPid = df[['account_id']]
#df_original = df_original.drop(columns=["pid_col"])

df_amt_dur = Synthesizer(df[['amount','duration']], pids=df[['account_id']]).sample()
print("syn amount/duration", scipy.stats.spearmanr(df_amt_dur['amount'], df_amt_dur['duration']))
