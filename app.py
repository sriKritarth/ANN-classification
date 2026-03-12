import io
import os
import tempfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


st.set_page_config(page_title="Power Plant ANN Regression", page_icon="⚡", layout="centered")


TARGET_COLUMN = "PE"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42


class PowerPlantNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)


@dataclass
class TrainingArtifacts:
    model: nn.Module
    scaler: StandardScaler
    feature_columns: list
    X_test: pd.DataFrame
    y_test: pd.Series
    test_predictions: pd.Series
    train_losses: list
    val_losses: list
    test_mse: float
    test_r2: float


def train_model(df: pd.DataFrame, epochs: int, batch_size: int, learning_rate: float) -> TrainingArtifacts:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PowerPlantNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_model_state = None
    best_model_loss = float("inf")

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_model_loss:
            best_model_loss = epoch_val_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_pred_tensor = model(X_test_tensor)

    test_predictions = pd.Series(test_pred_tensor.numpy().reshape(-1), index=X_test.index, name="Predicted PE")
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        feature_columns=list(X.columns),
        X_test=X_test,
        y_test=y_test,
        test_predictions=test_predictions,
        train_losses=train_losses,
        val_losses=val_losses,
        test_mse=float(test_mse),
        test_r2=float(test_r2),
    )


def make_loss_plot(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_prediction_table(artifacts: TrainingArtifacts) -> pd.DataFrame:
    out = artifacts.X_test.copy()
    out["Actual PE"] = artifacts.y_test
    out["Predicted PE"] = artifacts.test_predictions
    out["Error"] = out["Actual PE"] - out["Predicted PE"]
    return out.reset_index(drop=True)


def save_model_bundle(artifacts: TrainingArtifacts) -> bytes:
    payload = {
        "model_state_dict": artifacts.model.state_dict(),
        "scaler_mean": artifacts.scaler.mean_.tolist(),
        "scaler_scale": artifacts.scaler.scale_.tolist(),
        "feature_columns": artifacts.feature_columns,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return buffer.read()


st.title("⚡ Power Plant ANN Regression")
st.caption("Minimal Streamlit UI based on your notebook")

with st.sidebar:
    st.subheader("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    epochs = st.slider("Epochs", min_value=10, max_value=300, value=DEFAULT_EPOCHS, step=10)
    batch_size = st.selectbox("Batch size", options=[16, 32, 64, 128], index=1)
    learning_rate = st.selectbox("Learning rate", options=[0.1, 0.01, 0.001, 0.0001], index=2)
    train_clicked = st.button("Train model", type="primary", use_container_width=True)

st.markdown(
    f"Upload a dataset that contains **{TARGET_COLUMN}** as the target column. "
    "The app will train the same ANN architecture from the notebook and show metrics, a loss curve, and predictions."
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset preview")
        st.dataframe(data.head(), use_container_width=True)

        missing = data.isnull().sum()
        with st.expander("Missing values"):
            st.dataframe(missing.rename("null_count"))

        if TARGET_COLUMN not in data.columns:
            st.error(f"Target column '{TARGET_COLUMN}' was not found in the uploaded CSV.")
        elif not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            st.error("All columns must be numeric for this app.")
        elif train_clicked:
            with st.spinner("Training model..."):
                artifacts = train_model(data, epochs, batch_size, learning_rate)
                st.session_state["artifacts"] = artifacts
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

artifacts = st.session_state.get("artifacts")

if artifacts is not None:
    col1, col2 = st.columns(2)
    col1.metric("Test MSE", f"{artifacts.test_mse:.4f}")
    col2.metric("Test R²", f"{artifacts.test_r2:.4f}")

    st.subheader("Loss curve")
    st.pyplot(make_loss_plot(artifacts.train_losses, artifacts.val_losses), clear_figure=True)

    st.subheader("Test predictions")
    pred_df = build_prediction_table(artifacts)
    st.dataframe(pred_df.head(20), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download trained model bundle",
        data=save_model_bundle(artifacts),
        file_name="powerplant_model_bundle.pt",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.subheader("Manual prediction")
    with st.form("prediction_form"):
        values = []
        cols = st.columns(min(2, max(1, len(artifacts.feature_columns))))
        for idx, feature in enumerate(artifacts.feature_columns):
            with cols[idx % len(cols)]:
                default_value = float(artifacts.X_test[feature].mean())
                values.append(st.number_input(feature, value=default_value, format="%.4f"))
        submit_pred = st.form_submit_button("Predict")

    if submit_pred:
        sample = pd.DataFrame([values], columns=artifacts.feature_columns)
        sample_scaled = artifacts.scaler.transform(sample)
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_value = artifacts.model(sample_tensor).item()
        st.success(f"Predicted PE: {pred_value:.4f}")
else:
    st.info("Upload the CSV and click 'Train model' to start.")
