import streamlit as st
import warnings
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import calendar
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Ignore warnings
warnings.filterwarnings("ignore")

# Custom CSS for Full-Width Layout
st.set_page_config(layout="wide")
st.markdown(""" 
    <style> 
    .main { 
        background-color: #f7f7f7; 
    } 
    .stButton>button { 
        background-color: #FF6F61; 
        color: white; 
        border-radius: 10px; 
        font-size: 16px; 
    } 
    .big-title { 
        font-size: 36px; 
        font-weight: bold; 
        text-align: center; 
        color: #FF6F61; 
    } 
    .metric-box { 
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
    } 
    .explanation { 
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        margin-top: 10px; 
        border-left: 4px solid #FF6F61; 
    } 
    .insight-box { 
        background-color: #000000; 
        color: white !important; 
        padding: 15px; 
        border-radius: 10px; 
        margin-top: 10px; 
        border: 1px solid #FF6F61; 
    }
    .stDataFrame { 
        border: 1px solid #FF6F61;
        border-radius: 10px;
    }
    </style> 
""", unsafe_allow_html=True)


# Function to load data from Google Sheets
def load_data_from_google_sheets(url):
    csv_export_url = url.replace("/edit?usp=sharing", "/export?format=csv")
    data = pd.read_csv(csv_export_url)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    return data


def connect_to_google_sheets(sheet_name="Products"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    try:
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/12meeVmoFhLfmyaSucSETKRtfSAarRNKQox_HCF0uGQo/edit?usp=sharing")
        return sheet.worksheet(sheet_name)
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None


# CRUD Operations Functions
def create_data(sheet, item_code, item, price):
    try:
        sheet.append_row([item_code, item, price])
        st.success(f"✅ Created new product: {item} (Code: {item_code}) at ${price}")
    except Exception as e:
        st.error(f"❌ Error adding product: {e}")


def update_data(sheet, item, new_price):
    try:
        data = sheet.get_all_records()
        for i, row in enumerate(data, start=2):
            if row["Item"] == item:
                sheet.update_cell(i, 2, new_price)
                st.success(f"✅ Updated {item} price to ${new_price}")
                return
        st.error(f"❌ Product {item} not found.")
    except Exception as e:
        st.error(f"❌ Error updating product: {e}")


def delete_data(sheet, item):
    try:
        data = sheet.get_all_records()
        for i, row in enumerate(data, start=2):
            if row["Item"] == item:
                sheet.delete_rows(i)
                st.success(f"✅ Deleted product: {item}")
                return
        st.error(f"❌ Product {item} not found.")
    except Exception as e:
        st.error(f"❌ Error deleting product: {e}")


def perform_trend_decomposition(time_series, period=7):
    decomposition = seasonal_decompose(time_series, model='additive', period=period)

    # Calculate mean sales for the time series
    mean_sales = time_series.mean()

    # Generate insights
    insights = {
        'best_day': f"{time_series.idxmax().strftime('%A, %Y-%m-%d')} ({time_series.max():.0f} units)",
        'worst_day': f"{time_series.idxmin().strftime('%A, %Y-%m-%d')} ({time_series.min():.0f} units)",
        'trend_strength': decomposition.trend.max() - decomposition.trend.min(),
        'seasonal_impact': decomposition.seasonal.max() - decomposition.seasonal.min(),
        'residual_variability': decomposition.resid.std(),
        'peak_sales': [f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})" for date in
                       time_series[time_series > mean_sales].index],
        'decline_sales': [f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})" for date in
                          time_series[time_series < mean_sales].index],
        'yearly_comparison': {}
    }

    # Compare with previous years
    for year in time_series.index.year.unique():
        yearly_data = time_series[time_series.index.year == year]
        yearly_insights = {
            'best_day': f"{yearly_data.idxmax().strftime('%A, %Y-%m-%d')} ({yearly_data.max():.0f} units)",
            'worst_day': f"{yearly_data.idxmin().strftime('%A, %Y-%m-%d')} ({yearly_data.min():.0f} units)",
            'average_sales': yearly_data.mean(),
            'total_sales': yearly_data.sum()
        }
        insights['yearly_comparison'][year] = yearly_insights

    return decomposition, insights


def calculate_combined_error(data):
    status_text = st.empty()
    status_text.text("Aggregating sales data for all items...")

    # Aggregate all sales by date based on QUANTITY
    date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
    combined_time_series = data.groupby('Date')['Quantity'].sum().reindex(date_range, fill_value=0)
    time_series_smoothed = combined_time_series.rolling(window=7, center=True, min_periods=1).mean()

    # Train-test split (last 30 days)
    train_size = len(time_series_smoothed) - 30
    if train_size < 30:
        st.error("Insufficient data to perform a reliable test. At least 60 days of data are needed.")
        return None, None, None, None

    train, test = time_series_smoothed[:train_size], time_series_smoothed[train_size:]
    train_log = np.log1p(train)

    try:
        status_text.text("Building forecast model...")
        model = auto_arima(train_log, seasonal=True, m=7, suppress_warnings=True, error_action='ignore', stepwise=True,
                           n_jobs=-1)

        # --- 1. Standard Test ---
        status_text.text("Performing standard test...")
        standard_forecast_log = model.predict(n_periods=len(test))
        standard_forecast = np.expm1(standard_forecast_log)
        standard_forecast[standard_forecast < 0] = 0
        standard_mae = mean_absolute_error(test, standard_forecast)
        standard_rmse = np.sqrt(mean_squared_error(test, standard_forecast))

        # --- 2. Dynamic Test ---
        status_text.text("Performing dynamic test (this may take a moment)...")
        history = list(train_log)
        dynamic_predictions = []
        for t in range(len(test)):
            temp_model = ARIMA(history, order=model.order, seasonal_order=model.seasonal_order)
            model_fit = temp_model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            dynamic_predictions.append(yhat)
            history.append(np.log1p(test.iloc[t]))  # Add actual observed value

        dynamic_forecast = np.expm1(dynamic_predictions)
        dynamic_forecast[dynamic_forecast < 0] = 0
        dynamic_mae = mean_absolute_error(test, dynamic_forecast)
        dynamic_rmse = np.sqrt(mean_squared_error(test, dynamic_forecast))

        status_text.empty()
        return standard_mae, standard_rmse, dynamic_mae, dynamic_rmse

    except Exception as e:
        st.error(f"An error occurred during model building: {e}")
        status_text.empty()
        return None, None, None, None


# Load Data
data = load_data_from_google_sheets(
    "https://docs.google.com/spreadsheets/d/12meeVmoFhLfmyaSucSETKRtfSAarRNKQox_HCF0uGQo/edit?usp=sharing")
data = data.dropna(subset=['Date'])

# Sidebar Configuration
st.sidebar.title("🛠 CRUD Operations")
action = st.sidebar.selectbox("Select Action", ["Create", "Update", "Delete"])

# New sidebar controls for showing/hiding sections
st.sidebar.header("📊 Display Options")
show_combined_performance = st.sidebar.checkbox("Show Combined Performance for All Items", value=False)
show_standard_dynamic_test = st.sidebar.checkbox("Show Standard & Dynamic Test Results", value=False)

# CRUD Operations
sheet = connect_to_google_sheets()
if action == "Create":
    st.sidebar.subheader("Create New Product")
    new_item_code = st.sidebar.text_input("Item Code")
    new_item = st.sidebar.text_input("Product Name")
    new_price = st.sidebar.number_input("Price", min_value=0.0, value=0.0)
    if st.sidebar.button("Create Product"):
        create_data(sheet, new_item_code, new_item, new_price)

elif action == "Update":
    st.sidebar.subheader("Update Product")
    product_names = [row["Item"] for row in sheet.get_all_records()]
    item_to_update = st.sidebar.selectbox("Select Product to Update", product_names)
    new_price = st.sidebar.number_input("New Price", min_value=0.0, value=0.0)
    if st.sidebar.button("Update Product"):
        update_data(sheet, item_to_update, new_price)

elif action == "Delete":
    st.sidebar.subheader("Delete Product")
    product_names = [row["Item"] for row in sheet.get_all_records()]
    item_to_delete = st.sidebar.selectbox("Select Product to Delete", product_names)
    if st.sidebar.button("Delete Product"):
        delete_data(sheet, item_to_delete)

# Main Dashboard
st.markdown('<p class="big-title">🍞 Bakery Sales Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Track sales, forecast trends, and optimize product demand!")

# Sidebar Controls
st.sidebar.header("🔍 Analysis Controls")
products = ["Choose Product"] + sorted(data['Item'].unique())
selected_product = st.sidebar.selectbox("Select Product", products)

min_date = data['Date'].min()
max_date = data['Date'].max()

# Create tabs
tab1, tab2 = st.tabs(["📊 Sales & Forecasting", "🔗 Product Associations"])

# Tab 1: Sales & Forecasting
with tab1:
    st.header("📊 Business Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products Sold", f"{data['Quantity'].sum():,}")
    col2.metric("Unique Products", data['Item'].nunique())
    col3.metric("Total Transactions", data['ticket_number'].nunique())

    st.header("🏆 Top Performing Products")
    top_products = data.groupby('Item')['Quantity'].sum().nlargest(5).reset_index()

    # Create horizontal bar chart
    fig = px.bar(top_products,
                 x='Quantity',
                 y='Item',
                 color='Quantity',
                 orientation='h',
                 title="Top 5 Best-Selling Items",
                 labels={'Quantity': 'Units Sold', 'Item': 'Product'},
                 color_continuous_scale='mint')

    # Improve layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        margin=dict(l=120, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- SECTION FOR COMBINED MODEL PERFORMANCE (STANDARD & DYNAMIC) ---
    if show_combined_performance:
        st.header("📈 Combined Performance for All Items (by Quantity)")
        if show_standard_dynamic_test:
            st.markdown(
                "Calculate forecast accuracy using two methods: a **Standard Test** (forecasts all at once) and a **Dynamic Test** (forecasts day-by-day, updating with new data).")
            if st.button("Run Standard & Dynamic Tests"):
                with st.spinner("Running tests... The dynamic test may take a moment."):
                    s_mae, s_rmse, d_mae, d_rmse = calculate_combined_error(data)
                    if s_mae is not None:
                        st.subheader("Forecast Accuracy Metrics (Total Units Sold)")

                        st.markdown("#### Standard Test Results")
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric(label="MAE (Standard)", value=f"{s_mae:.2f} units")
                        mcol2.metric(label="RMSE (Standard)", value=f"{s_rmse:.2f} units")

                        st.markdown("#### Dynamic Test Results")
                        mcol3, mcol4 = st.columns(2)
                        mcol3.metric(label="MAE (Dynamic)", value=f"{d_mae:.2f} units")
                        mcol4.metric(label="RMSE (Dynamic)", value=f"{d_rmse:.2f} units")

                        st.markdown(f"""
                            <div class='insight-box'>
                                💡 **Accuracy Summary**
                                <ul>
                                    <li>The <b>Standard Test</b> shows the model's performance when predicting the next 30 days with no new information. The average error was <b>{s_mae:.2f} units</b> per day.</li>
                                    <li>The <b>Dynamic Test</b> simulates a real-world scenario by updating the forecast daily with the latest sales data. This resulted in an average error of <b>{d_mae:.2f} units</b> per day.</li>
                                    <li>A lower dynamic error suggests the model adapts well to new information.</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.markdown("Standard & Dynamic test results are currently hidden. Enable them in the sidebar to view.")

    # Improved SARIMA Forecasting Section
    if selected_product != "Choose Product":
        st.header("🔮 Advanced Forecasting (Optimized)")
        data['Total Sales'] = data['Quantity'] * data['Price']

        # Normalize the 'Date' column to ensure consistency
        data['Date'] = pd.to_datetime(data['Date']).dt.normalize()

        filtered_data = data[data['Item'] == selected_product].copy()

        # Group data by date and create the time series
        time_series_qty = filtered_data.groupby('Date')['Quantity'].sum()
        time_series_rev = filtered_data.groupby('Date')['Total Sales'].sum()

        # If a product has very few sales, it cannot be forecasted.
        if len(time_series_qty) < 40:  # Increased minimum data points for reliability
            st.warning(
                f"⚠️ Not enough data for '{selected_product}' to generate a reliable forecast. At least 40 sales days are recommended.")
        else:
            # Apply 7-day rolling mean (centered) for smoothing
            time_series_qty_smoothed = time_series_qty.rolling(window=7, center=True, min_periods=1).mean()
            time_series_rev_smoothed = time_series_rev.rolling(window=7, center=True, min_periods=1).mean()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Revenue Forecast")
            try:
                # Train-test split (last 30 days as test set)
                train_size = len(time_series_rev_smoothed) - 30
                train, test = time_series_rev_smoothed[:train_size], time_series_rev_smoothed[train_size:]

                # Automatic seasonality detection
                decomposition = seasonal_decompose(train, period=7)
                seasonal_strength = np.abs(decomposition.seasonal).mean() / decomposition.observed.std()

                # Determine if seasonal model is needed
                use_seasonal = seasonal_strength > 0.4  # Only use seasonal if strong seasonality

                # Log transform instead of Box-Cox (more stable for SARIMA)
                train_log = np.log1p(train)  # log(1+x) to handle zeros

                # Auto ARIMA with improved configuration
                model_rev = auto_arima(
                    train_log,
                    seasonal=use_seasonal,
                    m=7 if use_seasonal else 1,
                    start_p=1,
                    start_q=1,
                    max_p=3,
                    max_q=3,
                    d=1,  # Force at least one differencing
                    max_d=2,
                    start_P=0,
                    start_Q=0,
                    max_P=2,
                    max_Q=2,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    information_criterion='aic',
                    n_jobs=-1
                )

                st.write(f"Optimized SARIMA order: {model_rev.order}, seasonal order: {model_rev.seasonal_order}")

                # Forecast on test set
                test_forecast_log = model_rev.predict(n_periods=len(test))
                test_forecast = np.expm1(test_forecast_log)

                # Calculate evaluation metrics
                mae = mean_absolute_error(test, test_forecast)
                rmse = np.sqrt(mean_squared_error(test, test_forecast))

                # Dynamic forecasting for better test evaluation
                history = list(train_log)
                dynamic_forecast = []
                for t in range(len(test)):
                    model = ARIMA(history, order=model_rev.order, seasonal_order=model_rev.seasonal_order)
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    dynamic_forecast.append(output[0])
                    history.append(np.log1p(test.iloc[t]))

                dynamic_forecast = np.expm1(dynamic_forecast)
                dynamic_mae = mean_absolute_error(test, dynamic_forecast)
                dynamic_rmse = np.sqrt(mean_squared_error(test, dynamic_forecast))

                if show_standard_dynamic_test:
                    st.write(f"Standard Test Performance: MAE=₱{mae:.2f}, RMSE=₱{rmse:.2f}")
                    st.write(f"Dynamic Test Performance: MAE=₱{dynamic_mae:.2f}, RMSE=₱{dynamic_rmse:.2f}")

                # Final forecast with confidence intervals
                forecast_steps = st.slider("Forecast Days (Revenue)", 7, 31, 14, key="rev_forecast")

                # Refit model on full data
                full_model = ARIMA(np.log1p(time_series_rev_smoothed),
                                   order=model_rev.order,
                                   seasonal_order=model_rev.seasonal_order)
                full_model_fit = full_model.fit()

                # Get forecast with proper confidence intervals
                forecast_result = full_model_fit.get_forecast(steps=forecast_steps)
                forecast = np.expm1(forecast_result.predicted_mean)
                conf_int = np.expm1(forecast_result.conf_int())

                forecast_index = pd.date_range(time_series_rev_smoothed.index[-1], periods=forecast_steps + 1,
                                               freq='D')[1:]

                forecast_df = pd.DataFrame({
                    'date': forecast_index,
                    'mean': forecast,
                    'lower': conf_int.iloc[:, 0],
                    'upper': conf_int.iloc[:, 1]
                })

                # Plotting
                fig = go.Figure()

                # Historical data (last 60 days)
                hist_data = time_series_rev_smoothed.iloc[-60:]
                fig.add_trace(go.Scatter(
                    x=hist_data.index,
                    y=hist_data,
                    mode='lines',
                    name='Historical Revenue',
                    line=dict(color='#1f77b4')
                ))

                if show_standard_dynamic_test:
                    # Test data
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=test,
                        mode='lines',
                        name='Test Data',
                        line=dict(color='#9467bd')
                    ))

                    # Test forecast
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=test_forecast,
                        mode='lines',
                        name='Standard Forecast',
                        line=dict(color='#8c564b', dash='dot')
                    ))

                    # Dynamic forecast
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=dynamic_forecast,
                        mode='lines',
                        name='Dynamic Forecast',
                        line=dict(color='#7f7f7f', dash='dash')
                    ))

                # Future forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['mean'],
                    mode='lines',
                    name='Future Forecast',
                    line=dict(color='#ff7f0e', dash='dash')
                ))

                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                    y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,111,97,0.2)',
                    line_color='rgba(255,255,255,0)',
                    name='95% Confidence'
                ))

                fig.update_layout(
                    title=f"Revenue Forecast for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Sales (₱)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Forecast summary table
                st.subheader("📋 Forecast Summary")
                forecast_display = forecast_df[['date', 'mean']].copy()
                forecast_display.columns = ['Date', 'Forecasted Sales']
                forecast_display.set_index('Date', inplace=True)
                forecast_display = forecast_display.round(2)
                st.dataframe(
                    forecast_display.style.format({"Forecasted Sales": "₱{:.2f}"}),
                    height=300
                )

                # Generate insights
                first_date = forecast_df['date'].iloc[0].strftime('%b %d')
                last_date = forecast_df['date'].iloc[-1].strftime('%b %d')
                first_value = forecast_df['mean'].iloc[0]
                last_value = forecast_df['mean'].iloc[-1]

                if show_standard_dynamic_test:
                    insight_text = f"""
                        📈 **What This Means**
                        - Model accuracy on test data: MAE=₱{mae:.2f}, RMSE=₱{rmse:.2f}
                        - Dynamic forecast accuracy: MAE=₱{dynamic_mae:.2f}, RMSE=₱{dynamic_rmse:.2f}
                        - Expected sales around **₱{first_value:.2f}** on {first_date}
                        - Forecast shows {'growth' if last_value > first_value else 'decline'} through {last_date}
                        - Typical daily sales between **₱{forecast_df['mean'].min():.2f}** - **₱{forecast_df['mean'].max():.2f}**
                    """
                else:
                    insight_text = f"""
                        📈 **What This Means**
                        - Expected sales around **₱{first_value:.2f}** on {first_date}
                        - Forecast shows {'growth' if last_value > first_value else 'decline'} through {last_date}
                        - Typical daily sales between **₱{forecast_df['mean'].min():.2f}** - **₱{forecast_df['mean'].max():.2f}**
                    """

                st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Revenue forecast error: {str(e)}")
                st.error("Try selecting a different seasonality period or product with more data")

            with col2:
                st.subheader("📦 Quantity Forecast")
                try:
                    # Similar optimization for quantity forecast
                    # Train-test split
                    train_size = len(time_series_qty_smoothed) - 30
                    train, test = time_series_qty_smoothed[:train_size], time_series_qty_smoothed[train_size:]

                    # Automatic seasonality detection
                    decomposition = seasonal_decompose(train, period=7)
                    seasonal_strength = np.abs(decomposition.seasonal).mean() / decomposition.observed.std()
                    use_seasonal = seasonal_strength > 0.4

                    # Log transform
                    train_log = np.log1p(train)

                    # Auto ARIMA with improved configuration
                    model_qty = auto_arima(
                        train_log,
                        seasonal=use_seasonal,
                        m=7 if use_seasonal else 1,
                        start_p=1,
                        start_q=1,
                        max_p=3,
                        max_q=3,
                        d=1,
                        max_d=2,
                        start_P=0,
                        start_Q=0,
                        max_P=2,
                        max_Q=2,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        information_criterion='aic',
                        n_jobs=-1
                    )

                    st.write(f"Optimized SARIMA order: {model_qty.order}, seasonal order: {model_qty.seasonal_order}")

                    # Forecast evaluation
                    test_forecast_log = model_qty.predict(n_periods=len(test))
                    test_forecast = np.expm1(test_forecast_log)

                    # Dynamic forecasting
                    history = list(train_log)
                    dynamic_forecast = []
                    for t in range(len(test)):
                        model = ARIMA(history, order=model_qty.order, seasonal_order=model_qty.seasonal_order)
                        model_fit = model.fit()
                        output = model_fit.forecast()
                        dynamic_forecast.append(output[0])
                        history.append(np.log1p(test.iloc[t]))

                    dynamic_forecast = np.expm1(dynamic_forecast)

                    mae = mean_absolute_error(test, test_forecast)
                    rmse = np.sqrt(mean_squared_error(test, test_forecast))
                    dynamic_mae = mean_absolute_error(test, dynamic_forecast)
                    dynamic_rmse = np.sqrt(mean_squared_error(test, dynamic_forecast))

                    if show_standard_dynamic_test:
                        st.write(f"Standard Test Performance: MAE={mae:.2f} units, RMSE={rmse:.2f} units")
                        st.write(
                            f"Dynamic Test Performance: MAE={dynamic_mae:.2f} units, RMSE={dynamic_rmse:.2f} units")

                    # Final forecast
                    forecast_steps = st.slider("Forecast Days (Quantity)", 7, 31, 14, key="qty_forecast")

                    # Refit on full data
                    full_model = ARIMA(np.log1p(time_series_qty_smoothed),
                                       order=model_qty.order,
                                       seasonal_order=model_qty.seasonal_order)
                    full_model_fit = full_model.fit()

                    forecast_result = full_model_fit.get_forecast(steps=forecast_steps)
                    forecast = np.expm1(forecast_result.predicted_mean)
                    conf_int = np.expm1(forecast_result.conf_int())

                    forecast_index = pd.date_range(time_series_qty_smoothed.index[-1], periods=forecast_steps + 1,
                                                   freq='D')[1:]

                    forecast_df = pd.DataFrame({
                        'date': forecast_index,
                        'mean': forecast,
                        'lower': conf_int.iloc[:, 0],
                        'upper': conf_int.iloc[:, 1]
                    })

                    # Store forecast in session state
                    st.session_state.quantity_forecast = {
                        'forecast_df': forecast_df,
                        'generated_date': datetime.now().date()
                    }

                    # Historical data graph
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=time_series_qty_smoothed.index[-60:],
                        y=time_series_qty_smoothed[-60:],
                        mode='lines',
                        name='Historical Quantity',
                        line=dict(color='#2ca02c')
                    ))

                    if show_standard_dynamic_test:
                        fig.add_trace(go.Scatter(
                            x=test.index,
                            y=test,
                            mode='lines',
                            name='Test Data',
                            line=dict(color='#e377c2')
                        ))

                        fig.add_trace(go.Scatter(
                            x=test.index,
                            y=test_forecast,
                            mode='lines',
                            name='Standard Forecast',
                            line=dict(color='#7f7f7f', dash='dot')
                        ))

                        fig.add_trace(go.Scatter(
                            x=test.index,
                            y=dynamic_forecast,
                            mode='lines',
                            name='Dynamic Forecast',
                            line=dict(color='#d62728', dash='dash')
                        ))

                    # Future forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['mean'],
                        mode='lines',
                        name='Future Forecast',
                        line=dict(color='#d62728', dash='dash')
                    ))

                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                        y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(97,210,214,0.2)',
                        line_color='rgba(255,255,255,0)',
                        name='95% Confidence'
                    ))

                    fig.update_layout(
                        title=f"Quantity Forecast for {selected_product}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Units",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Forecast summary table
                    st.subheader("📋 Forecast Summary")
                    forecast_display = forecast_df[['date', 'mean']].copy()
                    forecast_display.columns = ['Date', 'Forecasted Units']
                    forecast_display.set_index('Date', inplace=True)
                    forecast_display = forecast_display.round(0).astype(int)
                    st.dataframe(
                        forecast_display.style.format({"Forecasted Units": "{:.0f} units"}),
                        height=300
                    )

                    # Generate simple insights
                    avg_units = forecast_df['mean'].mean().round(0)
                    peak_day = forecast_df.loc[forecast_df['mean'].idxmax()]

                    if show_standard_dynamic_test:
                        insight_text = f"""
                            📦 **What This Means**
                            - Model accuracy on test data: MAE={mae:.2f} units, RMSE={rmse:.2f} units  
                            - Dynamic forecast accuracy: MAE={dynamic_mae:.2f} units, RMSE={dynamic_rmse:.2f} units
                            - Expected to sell **{avg_units:.0f} units** daily on average  
                            - Highest demand predicted on **{peak_day['date'].strftime('%b %d')}** ({peak_day['mean']:.0f} units)  
                            - Daily forecasts range from **{forecast_df['mean'].min():.0f}** to **{forecast_df['mean'].max():.0f}** units  
                        """
                    else:
                        insight_text = f"""
                            📦 **What This Means**
                            - Expected to sell **{avg_units:.0f} units** daily on average
                            - Highest demand predicted on **{peak_day['date'].strftime('%b %d')}** ({peak_day['mean']:.0f} units)
                            - Daily forecasts range from **{forecast_df['mean'].min():.0f}** to **{forecast_df['mean'].max():.0f}** units
                        """

                    st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Quantity forecast error: {str(e)}")
                    st.error("Try selecting a different seasonality period or product with more data")

        # Sales Analysis Section
        st.header("📊 Product Sales Trend Analysis")
        current_month = datetime.now().strftime('%B')
        months = sorted(data['Date'].dt.strftime('%B').unique(),
                        key=lambda x: list(calendar.month_name).index(x))
        selected_month = st.selectbox("Select Month", months, index=months.index(current_month))

        if selected_product != "Choose Product":
            try:
                # Prepare data with enhanced formatting
                monthly_data = data[(data['Date'].dt.strftime('%B') == selected_month) &
                                    (data['Item'] == selected_product)]

                yearly_comparison = monthly_data.groupby(monthly_data['Date'].dt.year)['Quantity'] \
                    .sum() \
                    .reset_index() \
                    .rename(columns={'Date': 'Year'})

                yearly_comparison['Year'] = yearly_comparison['Year'].astype(str)
                yearly_comparison['Formatted_Qty'] = yearly_comparison['Quantity'].apply(lambda x: f"{x:,}")

                # Add forecast if viewing current month and forecast exists
                if (selected_month == current_month and
                        'quantity_forecast' in st.session_state and
                        st.session_state.quantity_forecast['generated_date'] == datetime.now().date()):

                    forecast_df = st.session_state.quantity_forecast['forecast_df']
                    current_month_num = datetime.now().month
                    current_year = datetime.now().year

                    # Get all forecasts for current month
                    monthly_forecast = forecast_df[
                        (forecast_df['date'].dt.month == current_month_num) &
                        (forecast_df['date'].dt.year == current_year)
                        ]

                    if not monthly_forecast.empty:
                        forecast_total = monthly_forecast['mean'].sum().round()
                        yearly_comparison = yearly_comparison.append({
                            'Year': f"{current_year} (Forecast)",
                            'Quantity': forecast_total,
                            'Formatted_Qty': f"{forecast_total:,.0f} (forecast)"
                        }, ignore_index=True)

                # Create the visualization figure
                fig = go.Figure()

                # Add bars - different color for forecast
                colors = ['#FFA15A' if '(Forecast)' in str(year) else '#4C78A8'
                          for year in yearly_comparison['Year']]

                fig.add_trace(go.Bar(
                    x=yearly_comparison['Year'],
                    y=yearly_comparison['Quantity'],
                    marker_color=colors,
                    text=yearly_comparison['Formatted_Qty'],
                    textposition='outside',
                    name='Sales'
                ))

                # Add trendline (only using historical data)
                historical_data = yearly_comparison[~yearly_comparison['Year'].str.contains("Forecast")]
                if len(historical_data) >= 2:
                    try:
                        x = np.arange(len(historical_data))
                        y = historical_data['Quantity']
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)

                        # Calculate trendline metrics
                        y_mean = np.mean(y)
                        ss_tot = np.sum((y - y_mean) ** 2)
                        ss_res = np.sum((y - p(x)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)

                        # Create trendline coordinates
                        trend_x = historical_data['Year'].tolist()
                        trend_y = p(x).tolist()

                        # Extend to forecast position if present
                        if "(Forecast)" in yearly_comparison['Year'].values:
                            trend_x.append(yearly_comparison['Year'].iloc[-1])
                            trend_y.append(p(len(historical_data)))

                        fig.add_trace(go.Scatter(
                            x=trend_x,
                            y=trend_y,
                            mode='lines+markers',
                            name='Sales Trend',
                            line=dict(
                                color='#E45756',
                                width=3,
                                shape='spline',
                                smoothing=0.6
                            ),
                            marker=dict(
                                size=10,
                                color='#E45756',
                                symbol='diamond',
                                line=dict(width=1.5, color='#2D3F50')
                            ),
                            hovertemplate="<b>Trend Value</b><br>%{y:.1f} units<extra></extra>"
                        ))

                        # Add trendline equation annotation
                        fig.add_annotation(
                            x=0.05,
                            y=0.95,
                            xref='paper',
                            yref='paper',
                            text=f"Trend: y = {z[0]:.1f}x + {z[1]:.1f}<br>R² = {r_squared:.2f}",
                            showarrow=False,
                            font=dict(size=12, color='#2D3F50'),
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='#2D3F50',
                            borderwidth=1
                        )

                    except Exception as e:
                        st.warning(f"Trend analysis limited: {str(e)}")

                # Add forecast annotation if present
                if "(Forecast)" in yearly_comparison['Year'].values:
                    fig.add_annotation(
                        x=yearly_comparison['Year'].iloc[-1],
                        y=yearly_comparison['Quantity'].iloc[-1],
                        text="Forecast",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color='#FF6F61', size=12)
                    )

                # Update layout
                fig.update_layout(
                    title=dict(
                        text=f"📈 {selected_product} Unit Sold in {selected_month}",
                        font=dict(size=20, color='#2D3F50'),
                        x=0.03,
                        y=0.93
                    ),
                    xaxis=dict(
                        title='Year',
                        type='category',
                        gridcolor='rgba(45,63,80,0.1)',
                        linecolor='#2D3F50',
                        title_font=dict(size=14, color='#2D3F50'),
                        tickfont=dict(size=12, color='#2D3F50')
                    ),
                    yaxis=dict(
                        title='Units Sold',
                        gridcolor='rgba(45,63,80,0.1)',
                        linecolor='#2D3F50',
                        title_font=dict(size=14, color='#2D3F50'),
                        tickfont=dict(size=12, color='#2D3F50'),
                        rangemode='tozero'
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(245,245,245,1)',
                    bargap=0.25,
                    hoverlabel=dict(
                        bgcolor='white',
                        font_size=12,
                        font_family='Arial',
                        font_color='#2D3F50'
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1,
                        font=dict(size=12, color='#2D3F50')
                    ),
                    margin=dict(l=20, r=20, t=100, b=20),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
                if len(yearly_comparison) >= 2:
                    try:
                        # Calculate key metrics
                        max_year = yearly_comparison.loc[yearly_comparison['Quantity'].idxmax(), 'Year']
                        max_sales = yearly_comparison['Quantity'].max()
                        min_year = yearly_comparison.loc[yearly_comparison['Quantity'].idxmin(), 'Year']
                        min_sales = yearly_comparison['Quantity'].min()
                        latest_sales = yearly_comparison.iloc[-1]['Quantity']
                        slope = z[0]
                        trend_strength = "strong" if r_squared >= 0.7 else "moderate" if r_squared >= 0.4 else "weak"
                        direction = "increasing" if slope > 0 else "decreasing"
                        annual_change = abs(slope)
                        pct_change = (annual_change / np.mean(y)) * 100

                        insights = f"""
                        <div style="padding:15px; background-color:#000000; border-radius:10px; margin-top:20px;">
                            <h4 style="color:white; border-bottom:2px solid #4C78A8; padding-bottom:8px;">📌 Key Insights ({selected_month} Sales)</h4>
                            <ul style="list-style-type:none; padding-left:0;">
                                <li>📈 <strong>Trend Analysis:</strong> {direction.capitalize()} at {annual_change:.1f} units/year ({pct_change:.1f}% annual change)</li>
                                <li>🏆 <strong>Peak Performance:</strong> {max_year} recorded highest sales ({max_sales:,} units)</li>
                                <li>📉 <strong>Lowest Sales:</strong> {min_year} had minimum sales ({min_sales:,} units)</li>
                                <li>🔮 <strong>Recent Performance:</strong> {yearly_comparison.iloc[-1]['Year']} sales at {latest_sales:,} units</li>
                            </ul>
                            <p style="color:white; font-size:0.9em; margin-top:10px;">
                            💡 Interpretation: The {trend_strength} {direction} trend suggests {
                        'growing demand' if direction == 'increasing' else
                        'potential market challenges' if direction == 'decreasing' else
                        'market stability'
                        } for {selected_product} in {selected_month}
                            </p>
                        </div>
                        """
                        st.markdown(insights, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Insights limited: {str(e)}")

                # Trend Decomposition Section
                st.subheader("📈 Trend Decomposition")
                try:
                    time_series = monthly_data.groupby('Date')['Quantity'].sum()
                    decomposition, insights = perform_trend_decomposition(time_series)

                    # Combine all components into a single DataFrame for plotting
                    decomposition_df = pd.DataFrame({
                        'Date': decomposition.observed.index,
                        'Observed': decomposition.observed,
                        'Trend': decomposition.trend,
                        'Seasonal': decomposition.seasonal,
                        'Residual': decomposition.resid
                    })

                    # Plot all components in a single graph using Plotly
                    fig = go.Figure()

                    # Add observed component
                    fig.add_trace(go.Scatter(
                        x=decomposition_df['Date'],
                        y=decomposition_df['Observed'],
                        mode='lines',
                        name='Observed',
                        line=dict(color='blue')
                    ))

                    # Add trend component
                    fig.add_trace(go.Scatter(
                        x=decomposition_df['Date'],
                        y=decomposition_df['Trend'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='green', dash='dash')
                    ))

                    # Add seasonal component
                    fig.add_trace(go.Scatter(
                        x=decomposition_df['Date'],
                        y=decomposition_df['Seasonal'],
                        mode='lines',
                        name='Seasonal',
                        line=dict(color='orange', dash='dot')
                    ))

                    # Add residual component
                    fig.add_trace(go.Scatter(
                        x=decomposition_df['Date'],
                        y=decomposition_df['Residual'],
                        mode='lines',
                        name='Residual',
                        line=dict(color='red', dash='dash')
                    ))

                    # Update layout
                    fig.update_layout(
                        title=f"Trend Decomposition for {selected_product} in {selected_month}",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode="x unified",
                        legend=dict(x=0.02, y=0.98),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )

                    # Display the graph
                    st.plotly_chart(fig, use_container_width=True)

                    # Display insights
                    st.subheader("🔍 Decomposition Insights")

                    # Create a container for the insights
                    with st.container():
                        st.markdown(f"""
                        <div class='insight-box'>
                            <h4>📌 Key Insights from Decomposition</h4>
                            <ul>
                                <li>📅 <strong>Best Day:</strong> {insights['best_day']}</li>
                                <li>📅 <strong>Worst Day:</strong> {insights['worst_day']}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        # Yearly comparison using Streamlit's native table
                        st.subheader("📊 Yearly Performance Comparison")

                        # Prepare yearly data
                        yearly_data = []
                        for year, yearly_insights in insights['yearly_comparison'].items():
                            yearly_data.append({
                                "Year": year,
                                "Best Day": yearly_insights['best_day'],
                                "Worst Day": yearly_insights['worst_day'],
                                "Avg Sales": f"{yearly_insights['average_sales']:.2f}",
                                "Total Sales": f"{yearly_insights['total_sales']:.2f}"
                            })

                        # Display as a styled table
                        if yearly_data:
                            yearly_df = pd.DataFrame(yearly_data)
                            st.dataframe(
                                yearly_df.style
                                .set_properties(**{
                                    'background-color': '#000000',
                                    'color': 'white',
                                    'border': '1px solid #FF6F61'
                                })
                                .set_table_styles([{
                                    'selector': 'th',
                                    'props': [
                                        ('background-color', '#FF6F61'),
                                        ('color', 'white'),
                                        ('font-weight', 'bold')
                                    ]
                                }])
                                .hide(axis='index'),
                                height=(len(yearly_df) * 35) + 38,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("No yearly comparison data available")
                except Exception as e:
                    st.error(f"Trend decomposition error: {str(e)}")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    else:
        st.info("ℹ️ Please select a product from the sidebar to view analysis")

# Tab 2: Product Associations
with tab2:
    st.header("🔗 Product Association Analysis")
    st.markdown("Discover which products are frequently purchased together to optimize bundling and placement.")

    # Date range selector
    assoc_date_range = st.date_input("📆 Select Date Range for Analysis", [min_date, max_date])

    if len(assoc_date_range) == 2:
        assoc_start, assoc_end = [pd.to_datetime(date) for date in assoc_date_range]
        assoc_data = data[(data['Date'] >= assoc_start) & (data['Date'] <= assoc_end)]

        if not assoc_data.empty:
            # Parameter controls
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider("📌 Minimum Support", 0.001, 0.1, 0.01, 0.001,
                                        help="Minimum frequency of items appearing together")
            with col2:
                min_lift = st.slider("🔗 Minimum Lift", 1.0, 10.0, 2.0, 0.1,
                                     help="Minimum strength of association between items")

            # Prepare transaction data
            transactions = assoc_data.groupby('ticket_number')['Item'].apply(list).tolist()

            # One-hot encode transactions
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            # Get frequent itemsets
            frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

                if not rules.empty:
                    # Convert frozensets to strings for display
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

                    # Show top rules
                    st.subheader("📋 Top Association Rules")
                    st.dataframe(
                        rules.sort_values('lift', ascending=False).head(10).reset_index(drop=True),
                        height=400
                    )

                    # Parallel categories plot
                    st.subheader("🔄 Product Association Network")
                    fig = px.parallel_categories(
                        rules.head(20),
                        dimensions=['antecedents', 'consequents'],
                        color='lift',
                        color_continuous_scale=px.colors.sequential.Inferno,
                        labels={
                            'antecedents': 'Products Bought',
                            'consequents': 'Often Bought With',
                            'lift': 'Association Strength'
                        },
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Item pair frequency table
                    st.subheader("📊 Frequently Bought Together")

                    # Get all item pairs from transactions
                    pair_counts = {}
                    for transaction in transactions:
                        items = list(set(transaction))  # Remove duplicates within a transaction
                        for pair in combinations(items, 2):
                            sorted_pair = tuple(sorted(pair))
                            pair_counts[sorted_pair] = pair_counts.get(sorted_pair, 0) + 1

                    # Convert to DataFrame
                    pair_df = pd.DataFrame(
                        [(', '.join(pair), count) for pair, count in pair_counts.items()],
                        columns=['Item Pair', 'Count']
                    ).sort_values('Count', ascending=False)

                    st.dataframe(
                        pair_df.head(20).reset_index(drop=True),
                        height=400
                    )

                    # Generate insights
                    strongest_rule = rules.iloc[0]
                    insight_text = f"""
                        <div class='insight-box'>
                            <h4>💡 Association Insights</h4>
                            <ul>
                                <li>When customers buy <strong>{strongest_rule['antecedents']}</strong>, 
                                they're <strong>{strongest_rule['confidence'] * 100:.1f}%</strong> likely to also buy 
                                <strong>{strongest_rule['consequents']}</strong></li>
                                <li>This combination occurs in <strong>{strongest_rule['support'] * 100:.1f}%</strong> of transactions</li>
                                <li>The lift value of <strong>{strongest_rule['lift']:.2f}</strong> indicates a strong association</li>
                            </ul>
                            <p>Consider bundling these items or placing them near each other to increase sales!</p>
                        </div>
                    """
                    st.markdown(insight_text, unsafe_allow_html=True)
                else:
                    st.warning("No significant association rules found. Try adjusting the support or lift thresholds.")
            else:
                st.warning("No frequent itemsets found. Try adjusting the minimum support threshold.")
        else:
            st.warning("No data available for the selected date range")
    else:
        st.info("Please select a valid date range for analysis")