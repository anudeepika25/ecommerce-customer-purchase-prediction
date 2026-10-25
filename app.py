
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="E-commerce Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. MODERN DASHBOARD STYLING
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #17365d, #2563a6);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #e4edf9;
        font-size: 1rem;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 1rem;
        border-radius: 14px;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 1.2rem;
        border-radius: 16px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    .small-note {
        color: #8b95a5;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD THE TRAINED MODEL
# =========================================================
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    model = load_model(str(MODEL_PATH))
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


# =========================================================
# 4. TRAINING FEATURE SCHEMA
# =========================================================
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Location",
    "Device_Type",
    "Session_Duration",
    "Pages_Viewed",
    "Products_Viewed",
    "Cart_Added",
    "Previous_Purchases",
    "Total_Spending",
    "Discount_Used",
    "Category",
    "Sessions_Last_30_Days",
    "Customer_Engagement",
    "Pages_Per_Minute",
    "Products_Per_Page",
    "Avg_Spending_Per_Previous_Purchase",
    "Engagement_Level",
]

THRESHOLD = 0.35


# =========================================================
# 5. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🛍️ Purchase AI")
    st.caption("Customer behavior analytics")

    st.divider()

    st.subheader("About this project")
    st.write(
        "A machine learning application that estimates "
        "the likelihood of a customer making a purchase "
        "based on customer and browsing-session information."
    )

    st.subheader("Technology")
    st.markdown(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- Random Forest\n"
        "- Streamlit"
    )

    st.divider()

    st.subheader("Prediction settings")
    st.metric("Classification threshold", f"{THRESHOLD:.0%}")

    st.caption(
        "A probability at or above the threshold is classified "
        "as a predicted purchase."
    )

    st.divider()
    st.markdown(
        '<p class="small-note">Portfolio project · '
        'E-commerce analytics</p>',
        unsafe_allow_html=True,
    )


# =========================================================
# 6. MAIN HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🛍️ E-commerce Customer Purchase Prediction</h1>
        <p>
            Understand customer behavior and estimate purchase
            likelihood using a trained machine learning pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("The trained model could not be loaded.")
    st.code(model_error or "Unknown model loading error")
    st.info(
        "Run this application from your project's .venv "
        "environment and verify scikit-learn compatibility."
    )
    st.stop()

st.success("Trained model loaded successfully.", icon="✅")

st.markdown(
    "Enter the customer details below and select "
    "**Predict Purchase** to view the model's estimate."
)


# =========================================================
# 7. CUSTOMER INPUT FORM
# =========================================================
with st.form("customer_form"):
    st.subheader("👤 Customer Profile")
    profile_col1, profile_col2, profile_col3 = st.columns(3)

    with profile_col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=25,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
        )

    with profile_col2:
        location = st.selectbox(
            "Location",
            [
                "Bengaluru",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai",
                "Other",
                "Pune",
                "Vijayawada",
                "Visakhapatnam",
            ],
        )

        device_type = st.selectbox(
            "Device Type",
            ["Desktop", "Mobile", "Tablet"],
        )

    with profile_col3:
        category = st.selectbox(
            "Product Category",
            [
                "Accessories",
                "Beauty",
                "Books",
                "Electronics",
                "Fashion",
                "Grocery",
                "Home & Kitchen",
                "Sports",
            ],
        )

        previous_purchases = st.number_input(
            "Previous Purchases",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )

    st.divider()
    st.subheader("🧭 Browsing Session")

    session_col1, session_col2, session_col3 = st.columns(3)

    with session_col1:
        session_duration = st.number_input(
            "Session Duration (minutes)",
            min_value=1.0,
            max_value=76.42,
            value=15.0,
            step=0.5,
        )

        pages_viewed = st.number_input(
            "Pages Viewed",
            min_value=1,
            max_value=120,
            value=10,
            step=1,
        )

    with session_col2:
        products_viewed = st.number_input(
            "Products Viewed",
            min_value=1,
            max_value=60,
            value=12,
            step=1,
        )

        cart_added = st.selectbox(
            "Added to Cart?",
            ["No", "Yes"],
        )

    with session_col3:
        discount_used = st.selectbox(
            "Discount Used?",
            ["No", "Yes"],
        )

        sessions_last_30_days = st.number_input(
            "Sessions in Last 30 Days",
            min_value=0,
            max_value=16,
            value=5,
            step=1,
        )

    st.divider()
    st.subheader("💳 Customer Value & Engagement")

    value_col1, value_col2 = st.columns(2)

    with value_col1:
        total_spending = st.number_input(
            "Total Historical Spending",
            min_value=0.0,
            max_value=55856.16,
            value=5000.0,
            step=100.0,
        )

    with value_col2:
        customer_engagement = st.number_input(
            "Customer Engagement Score",
            min_value=6.5,
            max_value=100.0,
            value=60.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Purchase",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 8. FEATURE ENGINEERING AND PREDICTION
# =========================================================
if submitted:

    # Preserve the binary encoding used during training.
    cart_value = 1 if cart_added == "Yes" else 0
    discount_value = 1 if discount_used == "Yes" else 0

    # Recreate the same engineered features.
    pages_per_minute = pages_viewed / session_duration
    products_per_page = products_viewed / pages_viewed

    if previous_purchases > 0:
        avg_spending = total_spending / previous_purchases
    else:
        avg_spending = 0.0

    if customer_engagement <= 40:
        engagement_level = "Low"
    elif customer_engagement <= 70:
        engagement_level = "Medium"
    else:
        engagement_level = "High"

    # Create one record with all 18 features.
    input_values = {
        "Age": age,
        "Gender": gender,
        "Location": location,
        "Device_Type": device_type,
        "Session_Duration": session_duration,
        "Pages_Viewed": pages_viewed,
        "Products_Viewed": products_viewed,
        "Cart_Added": cart_value,
        "Previous_Purchases": previous_purchases,
        "Total_Spending": total_spending,
        "Discount_Used": discount_value,
        "Category": category,
        "Sessions_Last_30_Days": sessions_last_30_days,
        "Customer_Engagement": customer_engagement,
        "Pages_Per_Minute": pages_per_minute,
        "Products_Per_Page": products_per_page,
        "Avg_Spending_Per_Previous_Purchase": avg_spending,
        "Engagement_Level": engagement_level,
    }

    model_input = pd.DataFrame(
        [[input_values[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    st.divider()
    st.subheader("📋 Customer Input Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Customer Age", f"{age}")
    summary_col2.metric("Products Viewed", f"{products_viewed}")
    summary_col3.metric(
        "Cart Added",
        "Yes" if cart_value else "No",
    )

    with st.expander("View all 18 model input features"):
        st.dataframe(
            model_input,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("⚙️ Engineered Features")

    feature_col1, feature_col2, feature_col3, feature_col4 = (
        st.columns(4)
    )

    feature_col1.metric(
        "Pages / Minute",
        f"{pages_per_minute:.3f}",
    )
    feature_col2.metric(
        "Products / Page",
        f"{products_per_page:.3f}",
    )
    feature_col3.metric(
        "Average Spend / Purchase",
        f"{avg_spending:,.2f}",
    )
    feature_col4.metric(
        "Engagement Level",
        engagement_level,
    )

    # The saved pipeline already handles preprocessing.
    try:
        if list(model_input.columns) != FEATURE_COLUMNS:
            raise ValueError("Input feature names or order do not match.")

        probability = float(
            model.predict_proba(model_input)[0, 1]
        )
        predicted_purchase = int(probability >= THRESHOLD)

        st.divider()
        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns([1, 1.4])

        with result_col1:
            st.metric(
                "Estimated Purchase Probability",
                f"{probability:.1%}",
            )
            st.progress(probability)

        with result_col2:
            if predicted_purchase == 1:
                st.success(
                    "🛍️ Purchase likely",
                    icon="✅",
                )
                st.write(
                    "The estimated probability meets or exceeds "
                    "the configured classification threshold."
                )
            else:
                st.warning(
                    "Purchase less likely",
                    icon="ℹ️",
                )
                st.write(
                    "The estimated probability is below the "
                    "configured classification threshold."
                )

        st.caption(
            f"Classification threshold: {THRESHOLD:.0%}. "
            "A prediction is an estimate, not a guarantee."
        )

        st.subheader("💡 Business Interpretation")

        if cart_value == 1:
            st.write(
                "The customer has added a product to the cart. "
                "Cart activity is a useful purchase-intent signal "
                "in this project's model."
            )
        else:
            st.write(
                "No cart addition was reported. Consider reviewing "
                "the browsing experience and product information "
                "as possible areas for improvement."
            )

        if customer_engagement > 70:
            st.write(
                "The engagement score falls in the high-engagement "
                "band used by this project."
            )
        elif customer_engagement <= 40:
            st.write(
                "The engagement score falls in the low-engagement "
                "band used by this project."
            )
        else:
            st.write(
                "The engagement score falls in the medium-engagement "
                "band used by this project."
            )

        st.caption(
            "These observations describe input signals and model "
            "estimates; they do not establish causal relationships."
        )

    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.exception(exc)


# =========================================================
# 9. FOOTER
# =========================================================
st.divider()
st.caption(
    "E-commerce Customer Purchase Prediction | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
