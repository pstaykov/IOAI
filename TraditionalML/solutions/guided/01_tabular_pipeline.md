# Solutions: 01 · Tabular pipeline (guided)

One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).

## g1: stratified 80/20 split that keeps the survival ratio; random_state=SEED

Placeholder:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ____)
```
Answer:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
```

## g2: median imputation, then standardisation (two named steps)

Placeholder:
```python
____,
```
Answer:
```python
("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
```

## g3: one-hot encoder that does not crash on categories unseen during fit

Placeholder:
```python
("onehot", ____),
```
Answer:
```python
("onehot", OneHotEncoder(handle_unknown="ignore")),
```

## g4: ROC AUC needs the probability of the positive class, not hard labels

Placeholder:
```python
proba = ____
```
Answer:
```python
proba = clf.predict_proba(X_test)[:, 1]
```

## g5: 5-fold CV of clf on the TRAINING split, scored by ROC AUC

Placeholder:
```python
scores = ____
```
Answer:
```python
scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
```

## g6: grid keys for a Pipeline step are "<step name>__<parameter>"

Placeholder:
```python
grid = {____: [3, 5, None], ____: [0.05, 0.1]}
```
Answer:
```python
grid = {"model__max_depth": [3, 5, None], "model__learning_rate": [0.05, 0.1]}
```

## g7: macro-averaged F1 (mean of per-class F1, each class weighted equally)

Placeholder:
```python
f1 = ____
```
Answer:
```python
f1 = f1_score(y_test, pred, average="macro")
```

## g8: write `pred` as one integer per line, no header, no index

Placeholder:
```python
pd.Series(pred).to_csv("submissionA.csv", ____)
```
Answer:
```python
pd.Series(pred).to_csv("submissionA.csv", index=False, header=False)
```
