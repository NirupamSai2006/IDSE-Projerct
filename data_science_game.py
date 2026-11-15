"""Interactive game for discovering the bias-variance trade-off.

Run: streamlit run data_science_game.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


st.set_page_config(page_title="Model Tuning Challenge", page_icon="🎮", layout="wide")
st.title("🎮 Model Tuning Challenge")
st.write("Can you choose a model that learns the signal without memorizing noisy training data?")
with st.expander("How to play and what you'll learn", expanded=True):
    st.markdown("""
**Goal:** earn generalization points over four rounds by choosing a sample size and model complexity.

1. Choose how many noisy measurements to collect and a polynomial model degree.
2. Press **Run this round**. The app fits the model to those training measurements.
3. Compare training error with error on a separate hidden test sample. Higher degree can fit more shapes, but may overfit; more data can stabilize the fit.

**Learning objectives:** explain underfitting and overfitting; compare training and test error; describe how sample size affects model stability. The measurements are simulated from a smooth engineering-style signal plus noise. The test points are generated separately, and all scores are computed from the model each round.
""")


def data_for_round(round_number, sample_count):
    """Generate an independently seeded train/test sample from a noisy signal."""
    rng = np.random.default_rng(2031 + round_number * 113)
    x_train = rng.uniform(-1, 1, sample_count)
    x_test = rng.uniform(-1, 1, 350)
    signal = lambda x: np.sin(1.6 * np.pi * x) + 0.25 * x
    y_train = signal(x_train) + rng.normal(0, 0.22, sample_count)
    y_test = signal(x_test) + rng.normal(0, 0.22, len(x_test))
    return x_train, y_train, x_test, y_test, signal


def score_round(round_number, sample_count, degree):
    x_train, y_train, x_test, y_test, signal = data_for_round(round_number, sample_count)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                          StandardScaler(), LinearRegression())
    model.fit(x_train.reshape(-1, 1), y_train)
    train_pred = model.predict(x_train.reshape(-1, 1))
    test_pred = model.predict(x_test.reshape(-1, 1))
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    baseline = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
    points = max(0, min(100, round(100 * (1 - test_mse / baseline))))
    grid = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_train, y_train, s=22, alpha=0.65, label="Training measurements")
    ax.plot(grid, signal(grid), color="black", linestyle="--", label="Underlying signal")
    order = np.argsort(x_test)
    ax.plot(x_test[order], test_pred[order], color="#e45756", label=f"Fitted degree {degree}")
    ax.set(xlabel="Normalized sensor input", ylabel="Observed response", title="The red curve is fitted using training points")
    ax.legend()
    return points, train_mse, test_mse, baseline, fig


if "round_number" not in st.session_state:
    st.session_state.round_number = 1
    st.session_state.total_points = 0
    st.session_state.history = []
    st.session_state.result = None

if st.session_state.round_number <= 4:
    round_number = st.session_state.round_number
    st.subheader(f"Round {round_number} of 4 · Total points: {st.session_state.total_points}")
    left, right = st.columns(2)
    sample_count = left.select_slider("Training measurements", options=[12, 24, 50, 100], value=24)
    degree = right.select_slider("Polynomial degree (model complexity)", options=[1, 2, 3, 5, 8, 12], value=3)
    if st.button("Run this round", type="primary", key=f"run_{round_number}"):
        points, train_mse, test_mse, baseline, fig = score_round(round_number, sample_count, degree)
        st.session_state.result = {
            "round": round_number, "samples": sample_count, "degree": degree,
            "points": points, "train_mse": train_mse, "test_mse": test_mse,
            "baseline": baseline,
        }
        st.session_state.total_points += points
        st.session_state.history.append(st.session_state.result.copy())
        st.session_state.round_number += 1
        st.session_state.plot = fig
    if st.session_state.result and st.session_state.result["round"] == round_number:
        result = st.session_state.result
        st.pyplot(st.session_state.plot)
        a, b, c = st.columns(3)
        a.metric("Generalization points", f"{result['points']} / 100")
        b.metric("Training MSE", f"{result['train_mse']:.3f}")
        c.metric("Test MSE", f"{result['test_mse']:.3f}")
        if result["train_mse"] < result["test_mse"] * 0.45:
            st.warning("The model fits training data much better than unseen data: a sign of overfitting.")
        elif result["train_mse"] > result["baseline"] * 0.9:
            st.info("The model has not captured much of the signal yet: this may be underfitting.")
        else:
            st.success("Training and test errors are reasonably close for this round.")
        if st.button("Next round", key=f"next_{round_number}"):
            st.rerun()
else:
    st.subheader("Challenge complete!")
    st.metric("Total generalization points", f"{st.session_state.total_points} / 400")
    st.write("Compare your choices and errors. Did higher model complexity help every time? Did more measurements change the result?")
    st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)
    if st.button("Play again"):
        for key in ("round_number", "total_points", "history", "result", "plot"):
            st.session_state.pop(key, None)
        st.rerun()

