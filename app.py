
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Success Analytics",
    layout="wide"
)

st.title("Student Success Analytics")
st.caption(
    "Explore CSV data and predict numerical outcomes "
    "using multiple linear regression."
)

st.info(
    "This educational demonstration uses synthetic "
    "Kaggle data. Predictions are not validated for "
    "real-world student assessment."
)

# --------------------------------------------------
# 2. UPLOAD DATASET
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type="csv"
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("1. Dataset Preview")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.dataframe(df.head(), hide_index=True)

    # Identify numerical columns
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Exclude record identifiers
    numeric_cols = [
        c for c in numeric_cols
        if c.lower() not in ["id", "index"]
    ]

    if len(numeric_cols) < 3:
        st.error(
            "This app requires at least three "
            "numeric columns."
        )
        st.stop()

    # --------------------------------------------------
    # 3. DATA VISUALIZATION
    # --------------------------------------------------

    st.subheader("2. Explore Your Data")

    col1, col2 = st.columns(2)

    variable = col1.selectbox(
        "Choose a numeric column",
        numeric_cols
    )

    chart_type = col2.selectbox(
        "Choose a chart type",
        ["Histogram", "Scatter", "Line"]
    )

    if chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=variable,
            nbins=30,
            title=f"Distribution of {variable}"
        )

    elif chart_type == "Scatter":

        x_options = [
            c for c in numeric_cols
            if c != variable
        ]

        x_variable = st.selectbox(
            "Choose a column for the X-axis",
            x_options
        )

        plot_data = df[
            [x_variable, variable]
        ].dropna()

        # Sample for faster plotting
        if len(plot_data) > 5000:
            plot_data = plot_data.sample(
                n=5000,
                random_state=42
            )

        fig = px.scatter(
            plot_data,
            x=x_variable,
            y=variable,
            opacity=0.4,
            title=f"{variable} vs {x_variable}"
        )

    else:

        # Line chart uses row order, not time
        plot_data = df[
            [variable]
        ].dropna().head(200).reset_index(
            drop=True
        )

        plot_data["Row"] = range(
            1, len(plot_data) + 1
        )

        fig = px.line(
            plot_data,
            x="Row",
            y=variable,
            title=f"{variable} by Row Order"
        )

        st.caption(
            "The line connects observations in row "
            "order. This dataset is not a time series."
        )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------
    # 4. MULTIPLE LINEAR REGRESSION
    # --------------------------------------------------

    st.subheader("3. Multiple Linear Regression")

    # Default choices for our student dataset
    default_x1 = (
        "study_hours"
        if "study_hours" in numeric_cols
        else numeric_cols[0]
    )

    x1 = st.selectbox(
        "Input 1 (X1)",
        numeric_cols,
        index=numeric_cols.index(default_x1)
    )

    x2_options = [
        c for c in numeric_cols if c != x1
    ]

    default_x2 = (
        "class_attendance"
        if "class_attendance" in x2_options
        else x2_options[0]
    )

    x2 = st.selectbox(
        "Input 2 (X2)",
        x2_options,
        index=x2_options.index(default_x2)
    )

    y_options = [
        c for c in numeric_cols
        if c not in [x1, x2]
    ]

    default_y = (
        "exam_score"
        if "exam_score" in y_options
        else y_options[0]
    )

    target = st.selectbox(
        "Target (Y)",
        y_options,
        index=y_options.index(default_y)
    )

    # Keep only relevant columns and valid rows
    data = df[
        [x1, x2, target]
    ].dropna()

    if len(data) < 5:
        st.error(
            "At least 5 complete rows are needed."
        )
        st.stop()

    if data[x1].nunique() < 2 or \
       data[x2].nunique() < 2:
        st.error(
            "Each input must contain at least "
            "two distinct values."
        )
        st.stop()

    # Train model
    X = data[[x1, x2]]
    y = data[target]

    model = LinearRegression()
    model.fit(X, y)

    # Show model performance
    r2 = model.score(X, y)

    st.write("Regression Equation")

    st.code(
        f"Predicted {target} = "
        f"{model.intercept_:.3f} "
        f"+ ({model.coef_[0]:.3f} * {x1}) "
        f"+ ({model.coef_[1]:.3f} * {x2})"
    )

    st.metric(
        "Training R²",
        f"{r2:.3f}"
    )

    st.caption(
        "Training R² evaluates the fitted data, "
        "not unseen students."
    )

    # --------------------------------------------------
    # 5. INTERACTIVE PREDICTION
    # --------------------------------------------------

    st.subheader("4. Try a Prediction")

    input_col1, input_col2 = st.columns(2)

    value1 = input_col1.number_input(
        f"Enter {x1}",
        value=float(X[x1].median())
    )

    value2 = input_col2.number_input(
        f"Enter {x2}",
        value=float(X[x2].median())
    )

    # SAME COLUMN NAMES AND ORDER AS TRAINING
    new_data = pd.DataFrame(
        [[value1, value2]],
        columns=[x1, x2]
    )

    prediction = model.predict(
        new_data
    )[0]

    st.metric(
        f"Predicted {target}",
        f"{prediction:,.2f}"
    )

    # --------------------------------------------------
    # 6. MODEL LIMITATIONS
    # --------------------------------------------------

    for col, val in [
        (x1, value1),
        (x2, value2)
    ]:
        low = X[col].min()
        high = X[col].max()

        if val < low or val > high:
            st.warning(
                f"{col} is outside the training "
                f"range ({low:.2f} to {high:.2f}). "
                "Interpret the prediction cautiously."
            )

    if target == "exam_score":
        if prediction < 0 or prediction > 100:
            st.warning(
                "The predicted exam score is "
                "outside the usual 0-100 range."
            )

    st.subheader("5. Business Interpretation")

    st.write(
        "This model estimates numerical outcomes "
        "to support exploratory analysis; inaccurate "
        "predictions could lead to poor decisions "
        "or inappropriate resource allocation."
    )

    st.caption(
        "This application is an educational prototype. "
        "It should not be used for high-stakes decisions."
    )

else:
    st.info(
        "Upload a CSV file to begin exploring data "
        "and training a regression model."
    )
