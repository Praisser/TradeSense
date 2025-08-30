import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from core.feature_engineering import generate_features

# --- Load & preprocess ---
df = pd.read_csv("data/EURUSD_M15.csv", sep="\t")
df.columns = [col.strip().lower().replace('<', '').replace('>', '') for col in df.columns]
df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.drop(columns=['date'], inplace=True)
df.rename(columns={'tickvol': 'tick_volume'}, inplace=True)
df = generate_features(df)

# --- Features & Target ---
features = ['rsi', 'macd', 'ema_fast', 'ema_slow', 'atr']
X = df[features]
y = df['target']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# --- LightGBM Classifier ---
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

# --- Save model ---
dump(model, "models/forex_model.pkl")

# --- Accuracy ---
print("Train acc:", accuracy_score(y_train, model.predict(X_train)))
print("Test acc:", accuracy_score(y_test, model.predict(X_test)))
