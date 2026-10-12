
import streamlit as st
import pandas as pd
from pathlib import Path
from fraud_engine import detect_fraud

# Page configuration
st.set_page_config(
    page_title="Multi-Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Dashboard heading
st.title("🛡️ Multi-Fraud Detection System")
st.write(
    "Analyze financial transactions and identify "
    "potentially suspicious activity using machine learning."
)

st.divider()

# Load sample dataset
data_path = Path(__file__).parent / "data" / "sample_transactions.csv"

# Sidebar
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Choose a dataset to analyze.")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        transactions = pd.read_csv(uploaded_file)
        data_source = "Uploaded dataset"
    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()
else:
    transactions = pd.read_csv(data_path)
    data_source = "Sample dataset"

# Validate and display data
st.subheader("📊 Transaction Overview")
st.caption(f"Data source: {data_source}")

required_columns = [
    "transaction_id",
    "amount",
    "transaction_count",
    "account_age_days",
    "is_foreign"
]

missing_columns = [
    column for column in required_columns
    if column not in transactions.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )
    st.info(
        "Required columns: transaction_id, amount, "
        "transaction_count, account_age_days, is_foreign"
    )
    st.stop()

# Numeric conversion
for column in [
    "amount",
    "transaction_count",
    "account_age_days",
    "is_foreign"
]:
    transactions[column] = pd.to_numeric(
        transactions[column], errors="coerce"
    )

if transactions.empty:
    st.warning("The dataset contains no transactions.")
    st.stop()

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", len(transactions))

with col2:
    st.metric(
        "Total Transaction Amount",
        f"{transactions['amount'].sum():,.2f}"
    )

with col3:
    st.metric(
        "Foreign Transactions",
        int((transactions["is_foreign"] == 1).sum())
    )

with col4:
    st.metric(
        "New Accounts",
        int((transactions["account_age_days"] <= 7).sum())
    )

st.divider()

# Detect fraud button
st.subheader("🔍 Fraud Analysis")

if st.button("Run Fraud Detection", type="primary"):
    try:
        with st.spinner("Analyzing transactions..."):
            results = detect_fraud(transactions)

        st.success("Analysis completed!")

        fraud_count = int(results["fraud_flag"].sum())
        review_count = int(results["review_flag"].sum())

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Potentially Suspicious",
            fraud_count
        )

        metric2.metric(
            "Flagged for Review",
            review_count
        )

        st.subheader("📋 Analysis Results")

        # Filter options
        filter_choice = st.selectbox(
            "Choose transactions to display",
            [
                "All Transactions",
                "Potentially Suspicious",
                "Flagged for Review",
                "Not Flagged"
            ]
        )

        if filter_choice == "Potentially Suspicious":
            displayed = results[results["fraud_flag"]]
        elif filter_choice == "Flagged for Review":
            displayed = results[results["review_flag"]]
        elif filter_choice == "Not Flagged":
            displayed = results[~results["review_flag"]]
        else:
            displayed = results

        st.dataframe(
            displayed,
            use_container_width=True,
            hide_index=True
        )

        # Chart
        st.subheader("💰 Transaction Amounts")
        chart_data = results.set_index("transaction_id")[
            ["amount"]
        ]
        st.bar_chart(chart_data)

        # Download results
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Analysis Results",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

        st.caption(
            "Important: A flagged transaction is not proof of fraud. "
            "Review each case before making a decision."
        )

    except Exception as error:
        st.error(f"Analysis failed: {error}")

else:
    st.info(
        "Click **Run Fraud Detection** to analyze your transactions."
    )

st.divider()
st.caption("Multi-Fraud Detection System | Educational Project")
