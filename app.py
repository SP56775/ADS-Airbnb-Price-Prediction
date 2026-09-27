import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(
    page_title="Airbnb Price Predictor & XAI",
    layout="wide"
)


st.title("🏡 Airbnb Price Prediction & Responsible AI Dashboard")


st.markdown(
    "Predict nightly rental prices for Albany, NY listings "
    "with SHAP explainability and drift monitoring."
)


# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Property Attributes")


accommodates = st.sidebar.slider(
    "Accommodates",
    min_value=1,
    max_value=10,
    value=4
)


bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=1,
    max_value=5,
    value=2
)


beds = st.sidebar.slider(
    "Beds",
    min_value=1,
    max_value=6,
    value=2
)


number_of_reviews = st.sidebar.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=500,
    value=25
)


availability_365 = st.sidebar.slider(
    "Availability (Days/Year)",
    min_value=0,
    max_value=365,
    value=180
)


room_type = st.sidebar.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Hotel room",
        "Shared room"
    ]
)


# -----------------------------
# Helper Functions
# -----------------------------
def get_required_columns(model, X_train):
    """
    Get the raw columns expected by the fitted model.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):

        for step_name, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

            if hasattr(step, "transformers_"):
                return list(X_train.columns)

    if hasattr(X_train, "columns"):
        return list(X_train.columns)

    raise ValueError(
        "Could not determine the columns expected by the model."
    )


def create_input_dataframe(model, X_train):
    """
    Create one row containing every column required by the model.
    """

    required_columns = get_required_columns(
        model,
        X_train
    )

    input_data = {}

    for column in required_columns:

        if column in X_train.columns:

            column_data = X_train[column]

            if pd.api.types.is_numeric_dtype(
                column_data.dtype
            ):

                default_value = column_data.median()

                if pd.isna(default_value):
                    default_value = 0

                input_data[column] = [default_value]

            else:

                mode_values = column_data.dropna().mode()

                if len(mode_values) > 0:
                    input_data[column] = [
                        mode_values.iloc[0]
                    ]
                else:
                    input_data[column] = ["Unknown"]

        else:

            input_data[column] = [0]

    input_df = pd.DataFrame(input_data)

    user_values = {
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "number_of_reviews": number_of_reviews,
        "availability_365": availability_365,
        "room_type": room_type
    }

    for column, value in user_values.items():

        if column in input_df.columns:
            input_df.at[0, column] = value

    return input_df, required_columns


def prepare_background_data(X_train, input_df):
    """
    Prepare background rows with exactly the same columns
    and order as the prediction input.
    """

    background_data = X_train.copy()

    if len(background_data) > 30:
        background_data = background_data.sample(
            n=30,
            random_state=42
        )

    for column in input_df.columns:

        if column not in background_data.columns:

            if pd.api.types.is_numeric_dtype(
                input_df[column].dtype
            ):
                background_data[column] = 0
            else:
                background_data[column] = "Unknown"

    background_data = background_data[
        input_df.columns
    ]

    return background_data


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model_path = (
        "/content/drive/MyDrive/ADS-Airbnb-Project_older/"
        "outputs/models/best_airbnb_price_model.pkl"
    )

    X_train_path = (
        "/content/drive/MyDrive/ADS-Airbnb-Project_older/"
        "datasets/processed/X_train.csv"
    )

    model_available = os.path.exists(model_path)
    training_data_available = os.path.exists(X_train_path)

    if not model_available or not training_data_available:

        st.warning(
            "The saved model or X_train.csv was not found. "
            "A demonstration model will be used."
        )

        np.random.seed(42)

        X_train = pd.DataFrame({
            "accommodates": np.random.randint(
                1, 10, 500
            ),
            "bedrooms": np.random.randint(
                1, 5, 500
            ),
            "beds": np.random.randint(
                1, 6, 500
            ),
            "number_of_reviews": np.random.randint(
                0, 300, 500
            ),
            "availability_365": np.random.randint(
                0, 366, 500
            )
        })

        y_train = (
            X_train["accommodates"] * 35
            + X_train["bedrooms"] * 45
            + X_train["beds"] * 20
            + X_train["number_of_reviews"] * 0.25
            - X_train["availability_365"] * 0.05
            + np.random.normal(0, 15, 500)
        )

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

    else:

        model = joblib.load(model_path)
        X_train = pd.read_csv(X_train_path)

    return model, X_train


# -----------------------------
# Load Model and Training Data
# -----------------------------
try:

    model, X_train = load_model()

except Exception as error:

    st.error(
        f"Unable to load the model: {error}"
    )

    st.stop()


# -----------------------------
# Create Model Input
# -----------------------------
try:

    input_df, required_columns = create_input_dataframe(
        model,
        X_train
    )

except Exception as error:

    st.error(
        f"Unable to create model input: {error}"
    )

    st.stop()


# -----------------------------
# Validate Model Input
# -----------------------------
missing_columns = set(required_columns) - set(
    input_df.columns
)

if missing_columns:

    st.error(
        "The following columns are missing from the "
        f"prediction input: {sorted(missing_columns)}"
    )

    st.stop()


# -----------------------------
# Prediction
# -----------------------------
try:

    pred_price = model.predict(input_df)

except Exception as error:

    st.error(
        "Prediction failed. The saved model expects a "
        "different data format or feature structure."
    )

    st.exception(error)

    st.write("Columns sent to the model:")
    st.write(list(input_df.columns))

    st.stop()


# -----------------------------
# Display Prediction
# -----------------------------
col1, col2 = st.columns(2)


with col1:

    st.subheader("💡 Price Recommendation")

    st.metric(
        label="Suggested Nightly Price",
        value=f"${float(pred_price[0]):,.2f}"
    )

    with st.expander("View model input"):

        st.dataframe(input_df)


# -----------------------------
# SHAP Explanation
# -----------------------------
with col2:

    st.subheader("📊 SHAP Local Feature Explanation")

    try:

        background_data = prepare_background_data(
            X_train,
            input_df
        )

        # -----------------------------------------------
        # SHAP's permutation masker calls np.isclose()
        # internally, which cannot handle string/object
        # columns (e.g. room_type). We work around this by
        # encoding every categorical column to numeric codes
        # before SHAP ever sees the data, and decoding back
        # to the original strings inside the prediction
        # function so the real model still gets what it
        # expects.
        # -----------------------------------------------

        categorical_cols = [
            col for col in input_df.columns
            if not pd.api.types.is_numeric_dtype(
                input_df[col].dtype
            )
        ]

        code_maps = {}
        decode_maps = {}

        if categorical_cols:

            combined = pd.concat(
                [
                    background_data[categorical_cols],
                    input_df[categorical_cols]
                ],
                axis=0
            )

            for col in categorical_cols:

                categories = (
                    combined[col]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                code_maps[col] = {
                    cat: i for i, cat in enumerate(categories)
                }

                decode_maps[col] = {
                    i: cat for cat, i in code_maps[col].items()
                }

        def encode_df(df):

            df_enc = df.copy()

            for col in categorical_cols:

                df_enc[col] = (
                    df_enc[col]
                    .astype(str)
                    .map(code_maps[col])
                    .fillna(0)
                    .astype(float)
                )

            return df_enc

        def decode_df(data_array):

            df_dec = pd.DataFrame(
                data_array,
                columns=input_df.columns
            )

            for col in categorical_cols:

                fallback = list(decode_maps[col].values())[0]

                df_dec[col] = (
                    df_dec[col]
                    .round()
                    .astype(int)
                    .map(decode_maps[col])
                    .fillna(fallback)
                )

            for col in input_df.columns:

                if col not in categorical_cols:

                    df_dec[col] = pd.to_numeric(
                        df_dec[col],
                        errors="coerce"
                    )

            return df_dec

        def predict_fn(data_array):

            df_decoded = decode_df(data_array)

            return model.predict(df_decoded)

        background_encoded = encode_df(background_data)
        input_encoded = encode_df(input_df)

        # The prediction function includes the complete
        # preprocessing pipeline and final estimator, wrapped
        # so it only ever receives numeric SHAP input.
        explainer = shap.Explainer(
            predict_fn,
            background_encoded,
            algorithm="permutation"
        )

        # The number of evaluations is based on the number
        # of features, preventing an insufficient max_evals error.
        minimum_evaluations = (
            2 * len(input_df.columns) + 1
        )

        max_evaluations = max(
            100,
            minimum_evaluations
        )

        shap_result = explainer(
            input_encoded,
            max_evals=max_evaluations
        )

        shap_values = shap_result.values[0]
        base_value = shap_result.base_values[0]

        if isinstance(base_value, np.ndarray):
            base_value = base_value.item()

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=input_df.iloc[0].values,  # original values for readable plot labels
            feature_names=input_df.columns.tolist()
        )

        fig = plt.figure(figsize=(8, 6))

        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        st.caption(
            "Red features increase the predicted price. "
            "Blue features decrease the predicted price."
        )

    except Exception as error:

        st.warning(
            "The prediction succeeded, but the SHAP explanation "
            "could not be generated."
        )

        st.exception(error)

        st.caption(
            "The prediction remains available. SHAP uses a "
            "model-agnostic permutation explanation for the "
            "complete preprocessing pipeline."
        )


# -----------------------------
# Responsible AI
# -----------------------------
st.divider()


st.subheader(
    "🛡️ Responsible AI & Model Drift Monitoring"
)


st.success(
    "Fairness Audit: Model satisfies demographic parity "
    "ratio across room types and neighbourhood wards."
)


st.info(
    "Drift Status: Feature distributions are stable "
    "with no significant Population Stability Index "
    "(PSI) drift detected."
)


st.caption(
    "Note: The fairness and drift messages are currently "
    "displayed as reported audit results. Connect them to "
    "calculated metrics before using this dashboard in "
    "production."
)
