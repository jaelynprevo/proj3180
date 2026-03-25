
import csv
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# 1.  LOAD REAL DATA
# ─────────────────────────────────────────────────────────────

JOBS = ["teacher", "health", "services", "at_home"]  # 'other' = baseline

def load_csv(filepath: str):
    """
    Load student-por.csv (semicolon-separated).
    Returns encoded feature matrix X, target y, and feature names.
    """
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)

    raw = [
        (r["school"], r["sex"], int(r["age"]), r["address"],
         r["famsize"], r["Pstatus"], int(r["Medu"]), int(r["Fedu"]),
         r["Mjob"], r["Fjob"], r["reason"], r["guardian"],
         int(r["traveltime"]), int(r["studytime"]), int(r["failures"]),
         r["schoolsup"], r["famsup"], r["paid"], r["activities"],
         r["nursery"], r["higher"], r["internet"], r["romantic"],
         int(r["famrel"]), int(r["freetime"]), int(r["goout"]),
         int(r["Dalc"]), int(r["Walc"]), int(r["health"]),
         int(r["absences"]), float(r["G1"]), float(r["G2"]))
        for r in rows
    ]
    y = np.array([float(r["G3"]) * 5 for r in rows])  # convert 0-20 to 0-100
    X, feature_names = encode_features(raw)
    return X, y, feature_names

# 2.  FEATURE ENCODING

REASONS   = ["home", "reputation", "course"]   # 'other' = baseline
GUARDIANS = ["mother", "father"]                # 'other' = baseline

def encode_features(raw):
    """
    Convert raw tuples into a numeric matrix.
      Binary categoricals -> 0 / 1
      Ordinal integers    -> kept as-is
      Nominal             -> one-hot (baseline = 'other')
    """
    rows = []
    for (school, sex, age, address, famsize, pstatus, Medu, Fedu,
         Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
         schoolsup, famsup, paid, activities, nursery, higher, internet,
         romantic, famrel, freetime, goout, Dalc, Walc, health,
         absences, G1, G2) in raw:

        row = [
            1.0 if school  == "GP"  else 0.0,
            1.0 if sex     == "F"   else 0.0,
            float(age),
            1.0 if address == "U"   else 0.0,
            1.0 if famsize == "GT3" else 0.0,
            1.0 if pstatus == "T"   else 0.0,
            float(Medu),
            float(Fedu),
        ]
        for j in JOBS:
            row.append(1.0 if Mjob == j else 0.0)
        for j in JOBS:
            row.append(1.0 if Fjob == j else 0.0)
        for res in REASONS:
            row.append(1.0 if reason == res else 0.0)
        for g in GUARDIANS:
            row.append(1.0 if guardian == g else 0.0)
        row += [float(traveltime), float(studytime), float(failures)]
        for val in [schoolsup, famsup, paid, activities,
                    nursery, higher, internet, romantic]:
            row.append(1.0 if val == "yes" else 0.0)
        row += [
            float(famrel), float(freetime), float(goout),
            float(Dalc), float(Walc), float(health),
            float(absences), float(G1) * 5, float(G2) * 5,
        ]
        rows.append(row)

    feature_names = (
        ["school_GP", "sex_F", "age", "address_U", "famsize_GT3", "Pstatus_T", "Medu", "Fedu"]
        + [f"Mjob_{j}" for j in JOBS]
        + [f"Fjob_{j}" for j in JOBS]
        + [f"reason_{r}" for r in REASONS]
        + [f"guardian_{g}" for g in GUARDIANS]
        + ["traveltime", "studytime", "failures"]
        + ["schoolsup", "famsup", "paid", "activities",
           "nursery", "higher", "internet", "romantic"]
        + ["famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
    )
    return np.array(rows, dtype=float), feature_names

# 3.  PRE-PROCESSING

def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]

def standardize(X_train, X_test):
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])

# 4.  MODELS

def fit_normal_equation(X, y):
    """ w = (XᵀX)⁻¹ Xᵀy """
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def fit_gradient_descent(X, y, lr=0.05, n_epochs=1000):
    """
    Batch gradient descent:
      w ← w − lr · (1/n) Xᵀ(Xw − y)
    """
    n, p = X.shape
    w, losses = np.zeros(p), []
    for _ in range(n_epochs):
        resid  = X @ w - y
        w     -= lr * (X.T @ resid) / n
        losses.append(float(np.mean(resid ** 2)))
    return w, losses

def predict(X, w):
    return X @ w

# 5.  METRICS

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

# 6.  PLOTS

def plot_predictions(y_true, y_pred_ols, y_pred_gd):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predicted vs Actual Final Grade (G3, scale 0–100)", fontsize=13, fontweight="bold")
    for ax, y_pred, label, color in zip(
        axes,
        [y_pred_ols, y_pred_gd],
        ["Normal Equation (OLS)", "Gradient Descent"],
        ["steelblue", "tomato"],
    ):
        ax.scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors="white", s=60)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect fit")
        ax.set_xlabel("Actual G3"); ax.set_ylabel("Predicted G3")
        ax.set_title(f"{label}  |  R² = {r2(y_true, y_pred):.4f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()

def plot_loss_curve(losses):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="tomato", lw=1.5)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Gradient Descent — Training Loss Curve")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

def plot_coefficients(w_ols, feature_names):
    coefs = w_ols[1:]
    idx   = np.argsort(np.abs(coefs))[::-1]
    names, vals = [feature_names[i] for i in idx], coefs[idx]
    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], vals[::-1], color=colors[::-1])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Coefficient (standardised features)")
    plt.title("Feature Importance — OLS Coefficients")
    plt.tight_layout()
    plt.savefig("coefficients.png", dpi=150)
    plt.show()

# 7.  MAIN

def main():
    # Load real data — make sure student-por.csv is in the same folder
    X, y, feature_names = load_csv("student-por.csv")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.2)
    X_tr_s, X_te_s, mu, sigma = standardize(X_tr, X_te)
    X_tr_b = add_bias(X_tr_s)
    X_te_b = add_bias(X_te_s)

    w_ols        = fit_normal_equation(X_tr_b, y_tr)
    w_gd, losses = fit_gradient_descent(X_tr_b, y_tr, lr=0.05, n_epochs=1000)

    y_pred_ols = predict(X_te_b, w_ols)
    y_pred_gd  = predict(X_te_b, w_gd)

    print("=" * 50)
    print("  STUDENT GRADE PREDICTOR — RESULTS")
    print("=" * 50)
    for label, w, yp in [
        ("Normal Equation (OLS)", w_ols, y_pred_ols),
        ("Gradient Descent",      w_gd,  y_pred_gd),
    ]:
        print(f"\n▸ {label}")
        print(f"  MSE : {mse(y_te, yp):.4f}")
        print(f"  R²  : {r2(y_te, yp):.4f}")

    print("\n▸ OLS Coefficients")
    print(f"  {'intercept':25s}: {w_ols[0]:.4f}")
    for name, coef in zip(feature_names, w_ols[1:]):
        print(f"  {name:25s}: {coef:.4f}")

    plot_predictions(y_te, y_pred_ols, y_pred_gd)
    plot_loss_curve(losses)
    plot_coefficients(w_ols, feature_names)


if __name__ == "__main__":
    main()
