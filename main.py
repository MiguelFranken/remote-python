import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, make_scorer, classification_report
from sklearn.svm import SVC

random_state = 358025

# Load classification dataset
df = pd.read_csv('flights_classifying.csv', index_col=0)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna()  # drop null rows
df = df[df['CANCELLED'] == 0]  # ignore cancelled flights
df = df.sample(frac=1, random_state=random_state)  # randomly order data points

# Target feature
y = df["REIMBURSMENT"]


# Transform SCHEDULED_DEPARTURE_CATEGORY with OrdninalEncoder as Morning < Afternoon < Evening < Night
def transform_scheduled_departure_category(df):
    df_ordinal = df.copy()
    categories = [['Morning', 'Afternoon', 'Evening', 'Night']]
    enc = OrdinalEncoder(categories=categories)
    df_ordinal["SCHEDULED_DEPARTURE_CATEGORY"] = enc.fit_transform(df_ordinal[["SCHEDULED_DEPARTURE_CATEGORY"]])
    return df_ordinal


X = df.copy()
X = transform_scheduled_departure_category(X)

# drop non-descriptive features
non_descriptive_features = [
    #'YEAR',
    'DAY',
    'DAY_YEARLY',
    'SCHEDULED_TIME',
    #'WEEK',
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

    #'DISTANCE',
    'SCHEDULED_ARRIVAL',
    'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT'
]
X.drop(non_descriptive_features, axis=1, inplace=True)

# one-hot encoding of categorical features
X = pd.get_dummies(X, columns=[
    "AIRLINE",
    # "ORIGIN_AIRPORT",
    # "DESTINATION_AIRPORT"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.85,
    random_state=random_state,
    stratify=y
)

print("Descriptive features:")
print(X_train.columns)

# scaling
scaler = RobustScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# either "pr_auc", "f1_macro" or "roc_auc"
scoring = "pr_auc"


# precision-recall AUC scorer
def pr_auc_scorer(y_true, pos_probs):
    precision, recall, _ = precision_recall_curve(y_true, pos_probs)
    return auc(recall, precision)


# no skill model, stratified random class predictions, f1_macro scoring
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


# no skill model, stratified random class predictions, roc auc scoring
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


# no skill model, stratified random class predictions, pr_auc scoring
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

X_train_10000, _, y_train_10000, _ = train_test_split(
    X_train, y_train,
    train_size=10000,
    random_state=random_state,
    stratify=y_train
)

tuned_parameters = [
    {
        "kernel": ["linear"],
        "C": [0.001, 0.01],
        "class_weight": ['balanced']
    },
    {
        "kernel": ["rbf", "sigmoid", "poly"],
        "C": [0.001, 0.01, 1, 10],
        "degree": [3, 5, 7, 9],
        "class_weight": ['balanced'],
        "gamma": ['scale', 'auto', 1e-2, 1e-3, 1e-4]
    },
]

print("# Tuning hyper-parameters")
print()

## here we choose the scoring method that can be specified by the scoring variable
if scoring == "f1_macro":
    print("Using f1_macro scoring..")
    clf = GridSearchCV(
        SVC(),  # Support Vector Machine (SVC)
        tuned_parameters,  # For each candidate specified by tuned_parameters a model is fitted and compared.
        scoring="f1_macro",
        cv=2,
        n_jobs=-1,  # for parallel computation
        verbose=3
    )
elif scoring == "roc_auc":
    print("Using ROC_AUC scoring..")
    clf = GridSearchCV(
        SVC(),
        tuned_parameters,
        scoring="roc_auc",
        n_jobs=-1,
        cv=2,
        verbose=3
    )
elif scoring == "pr_auc":
    print("Using PR_AUC scoring..")
    clf = GridSearchCV(
        SVC(
            probability=True,  # PR AUC needs the probabilities. Google when necessary.
            random_state=random_state  # To make calculating the probabilities deterministic
        ),
        tuned_parameters,
        scoring=make_scorer(pr_auc_scorer, needs_proba=True),
        # needs_proba to pass the computed probabilities to the scorer
        n_jobs=-1,
        # verbose=3,
        cv=4,
    )

clf.fit(X_train_10000, y_train_10000)

print("Best parameters set found on train set:")
print()
print(clf.best_params_)
print()
print(f"Grid scores ({scoring}) on train set:")
print()
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

X_test_fraction = X_test
y_test_fraction = y_test

# prediction
y_pred = clf.best_estimator_.predict(X_test_fraction)
y_pred_proba = clf.best_estimator_.predict_proba(X_test_fraction)

# calculate scores
svm_pr_auc_score = pr_auc_scorer(y_test_fraction, y_pred_proba[:, 1])
svm_roc_auc_score = roc_auc_score(y_test_fraction, y_pred_proba[:, 1])
svm_f1_score = f1_score(y_test_fraction, y_pred, average='macro')

print("Final evaluation report on test set:")
print()
print(classification_report(y_test_fraction, y_pred))
print()
print('SVM PR AUC score on test set: %.3f' % svm_pr_auc_score)
print('SVM ROC AUC score on test set: %.3f' % svm_roc_auc_score)
print('SVM f1_macro score on test set: %.3f' % svm_f1_score)
print()
print('PR AUC score difference to baseline on test set: %.3f' % (svm_pr_auc_score - baseline_pr_auc_score))
print('ROC AUC score difference to baseline on test set: %.3f' % (svm_roc_auc_score - baseline_roc_auc_score))
print('f1_macro score difference to baseline on test set: %.3f' % (svm_f1_score - baseline_f1_score))
