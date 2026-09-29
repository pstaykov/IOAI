# Reference implementation for drill: 01 · Tabular pipeline (drill)
# Percent-format cells; each block fills the drill cell with the same id.

# %% [s1] 1. Load and split ⏱ 8 min
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
df = fetch_openml("titanic", version=1, as_frame=True).frame.drop(columns=["boat", "body", "home.dest"])
df["family"] = df["sibsp"] + df["parch"] + 1
cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "family"]
X = df[cols].copy()
y = df["survived"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# %% [s2] 2. Preprocessing and model pipeline ⏱ 10 min
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
num = [c for c in X.columns if c not in ("sex", "embarked", "pclass")]
cat = ["sex", "embarked", "pclass"]
X_train[cat] = X_train[cat].astype(object); X_test[cat] = X_test[cat].astype(object)
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
], sparse_threshold=0)
clf = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)

# %% [s3] 3. Cross-validation helper ⏱ 8 min
def cv_auc(model, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return cross_val_score(clone(model), X, y, cv=cv, scoring="roc_auc").mean()

# %% [s4] 4. Hyperparameter search ⏱ 10 min
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
hgb = Pipeline([("prep", clone(prep)), ("model", HistGradientBoostingClassifier(random_state=0))])
search = GridSearchCV(hgb, {"model__max_depth": [3, None], "model__learning_rate": [0.05, 0.1],
                            "model__max_leaf_nodes": [15, 31]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)

# %% [s5] 5. Logistic regression from scratch ⏱ 12 min
def predict_proba_np(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))

def fit_logreg(X, y, lam=1e-2, lr=0.1, epochs=500):
    n, d = X.shape
    w, b, losses = np.zeros(d), 0.0, []
    for _ in range(epochs):
        p = predict_proba_np(X, w, b)
        eps = 1e-12
        losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + lam / 2 * w @ w)
        g = p - y
        w -= lr * (X.T @ g / n + lam * w)
        b -= lr * g.mean()
    return w, b, losses

# %% [s6] 6. Threshold for macro F1, and an IOAI-format file ⏱ 7 min
def best_threshold(y_true, proba):
    ts = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (proba >= t).astype(int), average="macro") for t in ts]
    return float(ts[int(np.argmax(scores))])

def write_submission(pred, path):
    pd.Series(np.asarray(pred).astype(int)).to_csv(path, index=False, header=False)
