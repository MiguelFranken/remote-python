import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve

random_state = 358025

# Load classification dataset
df = pd.read_csv('flights_classifying.csv', index_col=0)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna() # drop null rows
df = df[df['CANCELLED'] == 0] # ignore cancelled flights
df = df.sample(frac=1, random_state=random_state) # randomly order data points

# Target feature
y = df["REIMBURSMENT"]

## Descriptive features


## Transform SCHEDULED_DEPARTURE_CATEGORY with OrdninalEncoder as Morning < Afternoon < Evening < Night
def transform_scheduled_departure_category(df):
    df_ordinal = df.copy()
    categories = [['Morning', 'Afternoon', 'Evening', 'Night']]
    enc = OrdinalEncoder(categories=categories)
    df_ordinal["SCHEDULED_DEPARTURE_CATEGORY"] = enc.fit_transform(df_ordinal[["SCHEDULED_DEPARTURE_CATEGORY"]])
    return df_ordinal


X = df.copy()
X = transform_scheduled_departure_category(X)

# drop non-descriptive features
## all these are deleted
non_descriptive_features = [
    'YEAR', 'DAY',
    'DAY_YEARLY',
    'SCHEDULED_TIME', 'WEEK',
    'TAIL_NUMBER', 'FLIGHT_NUMBER',  # Unique identifiers
    'WHEELS_ON', 'WHEELS_OFF',
    'TAXI_IN', 'TAXI_OUT',
    'AIR_TIME', 'DIVERTED',
    # 'SCHEDULED_DEPARTURE_CATEGORY',
    'SCHEDULED_DEPARTURE',
    'CANCELLED', 'CANCELLATION_REASON',
    'AIR_SYSTEM_DELAY',
    'SECURITY_DELAY', 'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY',
    'DEPARTURE_DELAY', 'ARRIVAL_DELAY',
    'REIMBURSMENT',
    'DEPARTURE_TIME', 'ARRIVAL_TIME', 'ELAPSED_TIME',

    'DISTANCE', 'SCHEDULED_ARRIVAL',
]
X.drop(non_descriptive_features, axis=1, inplace=True)

# one-hot encoding of categorical features
X = pd.get_dummies(X, columns=[
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT"
])

# scaling
ft = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(ft, columns=X.columns, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.85,
    random_state=random_state,
    stratify=y
)

# either "pr_auc", "f1_macro" or "roc_auc"
scoring = "pr_auc"

# Precision-Recall AUC scorer
# Note: We only need to specify a custom method for PR AUC
#       as the other scoring alternatives can be specified by using a scoring string.
#       See below when model is specified and see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
def pr_auc_scorer(y_true, pos_probs):
    precision, recall, _ = precision_recall_curve(y_true, pos_probs)
    return auc(recall, precision)


## Sources:
## https://towardsdatascience.com/calculating-a-baseline-accuracy-for-a-classification-model-a4b342ceb88f
## https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
## https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators

## DummyClassifier
##
## Simple baseline to compare with other (real) classifiers
## Generates predictions by random guessing but respecting the training set’s class distribution
## Note that with all DummyClassifier strategies, the predict method completely ignores the input data!

## Alternative:
## - DummyRegressor
##   See https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor

## Note:
## For the f1_macro baseline model we could use cross-validation.
## This would be a TODO if we decide to use this metric.

## Open Questions:
## - Should we choose f1_macro, roc auc or pr_auc?
## - Which strategy for the DummyClassifier?

## no skill model, stratified random class predictions, f1_macro scoring
# f1_micro (majority = undelayed flights), f1_macro (minority = delayed flights)
def baseline_f1_macro():
    clf = DummyClassifier(
        strategy="stratified",
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    y_true, y_pred = y_test, clf.predict(X_test)
    score = f1_score(y_true, y_pred, average='macro')

    print('DummyClassifier f1_macro score on test set: %.3f' % score)
    return score


## no skill model, stratified random class predictions, roc auc scoring
def baseline_roc_auc():
    # fit
    baseline_clf = DummyClassifier(
        strategy="stratified",
        random_state=random_state
    )
    baseline_clf.fit(X_train, y_train)

    # predict
    y_true, y_pred, y_pred_proba = y_test, baseline_clf.predict(X_test), baseline_clf.predict_proba(X_test)

    # report
    score = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("DummyClassifier ROC AUC score on test set: %.3f" % score)
    return score


## no skill model, stratified random class predictions, pr_auc scoring
## Precision/Recall Curve AUC
def baseline_pr_auc():
    # no skill model, stratified random class predictions
    baseline_clf = DummyClassifier(
        strategy='stratified',
        random_state=random_state
    )
    baseline_clf.fit(X_train, y_train)
    y_true, y_pred_probs = y_test, baseline_clf.predict_proba(X_test)
    pos_probs = y_pred_probs[:, 1]

    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)
    auc_score = auc(recall, precision)
    print('DummyClassifier PR AUC score on test set: %.3f' % auc_score)
    return auc_score


baseline_f1_score = baseline_f1_macro()
baseline_roc_auc_score = baseline_roc_auc()
baseline_pr_auc_score = baseline_pr_auc()