import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

# ---------------------- Data Loading & Preprocessing ---------------------- #
# Load data (you can read the processed CSV/Excel file)
df = pd.read_csv("Book12.csv")
print("Data loaded successfully for dashboard")

# Clean up the 'Lead City' column to remove any extra spaces
df['Lead City'] = df['Lead City'].str.strip()

# Convert Sales Date to datetime (if available) and extract the month
if 'Sales Date' in df.columns:
    df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce')
    df['Month'] = df['Sales Date'].dt.to_period('M').astype(str)

# ---------------------- Basic Analysis ---------------------- #
# Calculate per-row conversion rates (not used in aggregated summary)
df['Conversion_Accepted'] = np.where(
    df['Leads Created'] != 0,
    df['Leads accepted'] / df['Leads Created'] * 100,
    0
)

df['Conversion_Installation'] = np.where(
    df['Leads accepted'] != 0,
    df['Installed'] / df['Leads accepted'] * 100,
    0
)

# ---------------------- Dashboard Title & Section 1 ---------------------- #
st.title("Excitel Customer Acquisition Dashboard")

# Section 1: Data Analysis Overview
st.header("Section 1: Probability and Normal Distributions")
st.write("""
In this section, we aggregate the conversion data by city.
Our goal is to calculate the overall conversion rates based on the total leads created, leads accepted, 
and installations completed for each city, rather than calculating row-by-row averages. This aggregated approach 
gives a more accurate representation of the conversion performance. The conversion rates are displayed with a 
percentage sign to clearly indicate that they are percentages.
""")

# Chart 1.0: Total Leads Summary by City (includes 'Installed')
st.subheader("Chart 1.0: Total Leads Summary by City")
st.write("""
This table shows the total number of Leads Created, Leads Accepted, and Installed for each city. 
It provides a straightforward view of the funnel volume at each stage before we look at conversion rates.
""")
total_leads_by_city = df.groupby('Lead City').agg({
    'Leads Created': 'sum',
    'Leads accepted': 'sum',
    'Installed': 'sum'
}).reset_index()
st.table(total_leads_by_city.style.hide(axis="index"))

# Chart 1.1: Aggregated Conversion Rates by City
st.subheader("Chart 1.1: Aggregated Conversion Rates by City")
conversion_summary = df.groupby('Lead City').agg({
    'Leads Created': 'sum',
    'Leads accepted': 'sum',
    'Installed': 'sum'
}).reset_index()

# Calculate aggregated conversion rates using the sums
conversion_summary['Acceptance Rate'] = np.where(
    conversion_summary['Leads Created'] != 0,
    conversion_summary['Leads accepted'] / conversion_summary['Leads Created'] * 100,
    0
)
conversion_summary['Installation Rate'] = np.where(
    conversion_summary['Leads accepted'] != 0,
    conversion_summary['Installed'] / conversion_summary['Leads accepted'] * 100,
    0
)
conversion_summary['Total Conversion Rate'] = np.where(
    conversion_summary['Leads Created'] != 0,
    conversion_summary['Installed'] / conversion_summary['Leads Created'] * 100,
    0
)

# Format conversion rates as percentages
conversion_summary['Acceptance Rate'] = conversion_summary['Acceptance Rate'].apply(lambda x: f"{x:.2f}%")
conversion_summary['Installation Rate'] = conversion_summary['Installation Rate'].apply(lambda x: f"{x:.2f}%")
conversion_summary['Total Conversion Rate'] = conversion_summary['Total Conversion Rate'].apply(lambda x: f"{x:.2f}%")

st.table(conversion_summary.style.hide(axis="index"))

# Chart 1.2: Leads Accepted Distribution by City (Histogram + Binned Frequency Table)
st.subheader("Chart 1.2: Leads Accepted Distribution by City")
st.write("""
This histogram shows the distribution of weekly 'Leads accepted' for a selected city. 
It helps identify whether the number of accepted leads is consistent each week and if there are outliers.
Below the chart, a horizontally formatted binned frequency table of 'Leads accepted' is provided for numeric interpretation.
""")

selected_city = st.selectbox("Select City for Leads Accepted Distribution", df['Lead City'].unique())
city_data = df[df['Lead City'] == selected_city]

# Histogram
fig_hist = px.histogram(
    city_data, 
    x="Leads accepted", 
    nbins=15,
    title=f"Leads Accepted Distribution for {selected_city}"
)
st.plotly_chart(fig_hist)

# Binned Frequency Table
bins = [0, 49, 99, 149, 199, 249, 299, float('inf')]
labels = ["0-49", "50-99", "100-149", "150-199", "200-249", "250-299", "300+"]

# Create a new column for binning
city_data['Accepted Bin'] = pd.cut(city_data['Leads accepted'], bins=bins, labels=labels, include_lowest=True)
bin_counts = city_data.groupby('Accepted Bin').size().reset_index(name='Count')
bin_counts_pivot = bin_counts.set_index('Accepted Bin').T
st.write(f"**Binned Frequency of 'Leads accepted' for {selected_city}**")
st.table(bin_counts_pivot.style.hide(axis='index'))

# Chart 1.3: Monthly Leads Accepted Comparison Across Cities (Line Chart)
st.subheader("Chart 1.3: Monthly Leads Accepted Comparison Across Cities")
st.write("""
This line chart compares the total number of leads accepted each month across different cities. 
It helps visualize trends over time and identify seasonal patterns or changes in performance.
""")
if 'Month' in df.columns:
    monthly_leads = df.groupby(['Month', 'Lead City'])['Leads accepted'].sum().reset_index()
    fig_line = px.line(monthly_leads, x='Month', y='Leads accepted', color='Lead City', markers=True,
                       title="Monthly Total Leads Accepted by City")
    st.plotly_chart(fig_line)
else:
    st.write("Sales Date column is not available to create a monthly comparison.")

# Chart 1.4: Monthly Conversion Rate Trend (Line Chart)
st.subheader("Chart 1.4: Monthly Conversion Rate Trend by City")
st.write("""
This chart shows the monthly conversion rate trend (calculated as total installed divided by total leads created) for each city.
It helps identify periods of high or low performance over time.
""")
if 'Month' in df.columns:
    monthly_conversion = df.groupby(['Month', 'Lead City']).agg({
        'Leads Created': 'sum',
        'Installed': 'sum'
    }).reset_index()
    monthly_conversion['Conversion Rate'] = np.where(
        monthly_conversion['Leads Created'] != 0,
        monthly_conversion['Installed'] / monthly_conversion['Leads Created'] * 100,
        0
    )
    fig_line_conversion = px.line(monthly_conversion, x='Month', y='Conversion Rate', color='Lead City', markers=True,
                                  title="Monthly Conversion Rate by City")
    fig_line_conversion.update_yaxes(title="Conversion Rate (%)")
    st.plotly_chart(fig_line_conversion)
else:
    st.write("Sales Date column is not available to create a monthly conversion trend chart.")

# Chart 1.5: City-Level Funnel Comparison (Grouped Bar Chart)
st.subheader("Chart 1.5: City-Level Funnel Comparison")
st.write("""
This grouped bar chart shows the total 'Leads Created', 'Leads accepted', and 'Installed' for each city.
It provides an at-a-glance comparison of how each city progresses leads through the funnel stages.
""")
funnel_data = df.groupby('Lead City').agg({
    'Leads Created': 'sum',
    'Leads accepted': 'sum',
    'Installed': 'sum'
}).reset_index()
funnel_data_melted = funnel_data.melt(id_vars='Lead City', var_name='Stage', value_name='Count')
fig_bar = px.bar(
    funnel_data_melted, x='Lead City', y='Count', color='Stage',
    barmode='group', title="Funnel by City (Created → Accepted → Installed)"
)
st.plotly_chart(fig_bar)

# ---------------------- Section 2: Descriptive Statistics ---------------------- #
st.header("Section 2: Descriptive Statistics")
st.write("""
In this section, we provide descriptive statistics related to the installations data for each city.
We compute the mean and standard deviation of weekly installations and determine the proportion of weeks 
where the installations fall within one standard deviation of the mean. 
This analysis helps assess the consistency and variability in installations across the cities.
""")

# Calculate descriptive statistics for 'Installed' for each city
desc_stats = df.groupby('Lead City').agg(
    Mean_Installations=('Installed', 'mean'),
    Std_Installations=('Installed', 'std'),
    Weeks=('Installed', 'count')
).reset_index()

# Define a function to calculate the proportion of weeks where installations are within one standard deviation
def calc_proportion(group):
    mean_val = group['Installed'].mean()
    std_val = group['Installed'].std()
    total = group.shape[0]
    count_within = group[(group['Installed'] >= (mean_val - std_val)) & (group['Installed'] <= (mean_val + std_val))].shape[0]
    proportion = (count_within / total) * 100  # percentage
    return proportion

prop_df = df.groupby('Lead City').apply(calc_proportion).reset_index(name='Proportion within 1 STD (%)')

# Merge the descriptive stats with the proportion data
desc_stats = desc_stats.merge(prop_df, on='Lead City')
desc_stats['Mean_Installations'] = desc_stats['Mean_Installations'].round(2)
desc_stats['Std_Installations'] = desc_stats['Std_Installations'].round(2)
desc_stats['Proportion within 1 STD (%)'] = desc_stats['Proportion within 1 STD (%)'].round(2)

# Display the descriptive statistics table
st.subheader("Chart 2.0: Descriptive Statistics Table")
st.table(desc_stats.style.hide(axis="index"))

# Chart 2.1: Mean Installations with Standard Deviation (Bar Chart with Error Bars)
st.subheader("Chart 2.1: Mean Installations by City with Standard Deviation")
fig_bar_desc = px.bar(desc_stats, x='Lead City', y='Mean_Installations', error_y='Std_Installations', 
                      title="Mean Weekly Installations by City with STD Error Bars",
                      labels={"Mean_Installations": "Mean Installations", "Lead City": "City"})
st.plotly_chart(fig_bar_desc)

# ---------------------- Section 3: Confidence Intervals ---------------------- #
st.header("Section 3: Confidence Intervals")
st.write("""
In this section, we calculate confidence intervals to assess the reliability and variability of the data.
We provide:
- The 90% confidence interval for the **average installation time** (from lead acceptance to installation) for each city.
- The 90% confidence interval for the **number of leads accepted per week**, assuming Excitel typically receives 100 leads per week.
""")

# Chart 3.0: Average Installation Time Confidence Intervals
if 'Installation Time' in df.columns:
    ci_installation_data = []
    for city, group in df.groupby('Lead City'):
        installation_times = group['Installation Time'].dropna()
        n = len(installation_times)
        if n > 1:
            mean_time = installation_times.mean()
            sem_time = stats.sem(installation_times)
            ci = stats.t.interval(0.90, n-1, loc=mean_time, scale=sem_time)
            ci_installation_data.append({
                'Lead City': city,
                'Mean Installation Time': round(mean_time, 2),
                'Lower 90% CI': round(ci[0], 2),
                'Upper 90% CI': round(ci[1], 2)
            })
    ci_installation_df = pd.DataFrame(ci_installation_data)
    st.subheader("Chart 3.0: Average Installation Time with 90% Confidence Intervals")
    st.table(ci_installation_df.style.hide(axis="index"))
else:
    st.write("Installation Time data is not available.")

# Chart 3.1: Weekly Lead Acceptance Confidence Intervals
# Calculate confidence intervals for weekly leads accepted using a binomial approach
# Assuming Excitel receives 100 leads per week on average per city
ci_leads_data = []
z = 1.645  # z-score for 90% confidence
for city, group in df.groupby('Lead City'):
    total_created = group['Leads Created'].sum()
    total_accepted = group['Leads accepted'].sum()
    if total_created > 0:
        p = total_accepted / total_created  # aggregated acceptance rate
    else:
        p = 0
    mean_accepted = 100 * p  # expected accepted leads if 100 leads are received
    se = np.sqrt(100 * p * (1-p))
    lower_bound = mean_accepted - z * se
    upper_bound = mean_accepted + z * se
    ci_leads_data.append({
        'Lead City': city,
        'Acceptance Rate (p)': round(p, 4),
        'Mean Weekly Accepted': round(mean_accepted, 2),
        'Lower 90% CI': round(lower_bound, 2),
        'Upper 90% CI': round(upper_bound, 2)
    })
ci_leads_df = pd.DataFrame(ci_leads_data)
st.subheader("Chart 3.1: Weekly Lead Acceptance 90% Confidence Intervals")
st.table(ci_leads_df.style.hide(axis="index"))

# ---------------------- Section 4: Hypothesis Testing ---------------------- #
st.header("Section 4: Hypothesis Testing")
st.write("""
In this section, we test whether one city performs significantly better than the others in terms of conversion rates.
**Null Hypothesis (H₀):** The mean conversion accepted is equal for all cities.
**Alternative Hypothesis (H₁):** At least one city has a different mean conversion accepted.
A one-way ANOVA test is performed on the per-row conversion accepted values to assess statistical significance.
""")

# Extract conversion data per city (only consider rows with non-zero Leads Created to avoid division by zero)
anova_data = [group['Conversion_Accepted'] for name, group in df[df['Leads Created'] != 0].groupby('Lead City')]
f_stat, p_value = stats.f_oneway(*anova_data)
st.write(f"**F-statistic:** {f_stat:.2f}")
st.write(f"**p-value:** {p_value:.4f}")
if p_value < 0.05:
    st.write("Result: The differences in conversion rates between cities are statistically significant (reject H₀).")
else:
    st.write("Result: There is no statistically significant difference in conversion rates between cities (fail to reject H₀).")

# ---------------------- Section 5: Correlation and Regression ---------------------- #
st.header("Section 5: Correlation and Regression")
st.write("""
This section examines the relationship between the number of Leads Created and Installed.
- **Correlation:** We calculate the Pearson correlation coefficient to measure the strength and direction of the linear relationship.
- **Regression:** We build a linear regression model to predict the number of installations based on the number of leads created.
""")

# Correlation Analysis
corr_coef, p_val_corr = stats.pearsonr(df['Leads Created'], df['Installed'])
st.write(f"**Pearson Correlation Coefficient (Leads Created vs. Installed):** {corr_coef:.2f}")
st.write(f"**p-value for correlation:** {p_val_corr:.4f}")
if p_val_corr < 0.05:
    st.write("The correlation is statistically significant.")
else:
    st.write("The correlation is not statistically significant.")

# Regression Analysis: Predicting Installations based on Leads Created
# Prepare the data
X = df['Leads Created']
y = df['Installed']
X = sm.add_constant(X)  # adding a constant for the intercept

# Fit the regression model
model = sm.OLS(y, X).fit()

st.subheader("Regression Model: Predicting Installations from Leads Created")

# 1. Display Key Coefficients in a Table
coeff_df = pd.DataFrame({
    'Variable': model.params.index,
    'Coefficient': model.params.values,
    'Std. Error': model.bse.values,
    't-statistic': model.tvalues,
    'p-value': model.pvalues
})
# Round for neatness
coeff_df[['Coefficient', 'Std. Error', 't-statistic', 'p-value']] = coeff_df[['Coefficient', 'Std. Error', 't-statistic', 'p-value']].round(4)

st.write("**Model Coefficients**")
st.table(coeff_df.style.hide(axis="index"))

# 2. Display Model Performance in a Table
perf_dict = {
    'R-squared': [model.rsquared],
    'Adj. R-squared': [model.rsquared_adj],
    'F-statistic': [model.fvalue],
    'Prob (F-statistic)': [model.f_pvalue],
    'No. Observations': [int(model.nobs)]
}
perf_df = pd.DataFrame(perf_dict).round(4)

st.write("**Model Performance**")
st.table(perf_df)

# 3. Optional: Still provide the full text summary in a collapsible section
with st.expander("See Full Regression Output"):
    st.text(model.summary())

# Scatter Plot with Regression Line
st.subheader("Chart 5.0: Scatter Plot with Regression Line")
fig_scatter = px.scatter(df, x="Leads Created", y="Installed", trendline="ols",
                         title="Relationship between Leads Created and Installed")
st.plotly_chart(fig_scatter)

st.subheader("Regression Analysis: Predicting The Future")
st.write("""
✅ Significance:
Our analysis demonstrates a strong correlation between the number of Leads Created and Installed. This indicates that increasing lead generation tends to drive more installations. However, it’s important to note that not every lead converts into an installation.

📌 Operational Insight:
While higher lead volumes generally contribute to more installations, the quality of these leads plays a critical role. Optimizing the conversion process throughout the funnel ensuring leads are properly nurtured and pre-qualified is essential to enhance overall performance.

🚀 Strategy Recommendations:
         
• Scale Lead Generation Campaigns:
    Expand your advertising channels, leverage referral programs, and develop strategic partnerships to boost the volume of quality leads.         
         
• Optimize the Conversion Funnel:
    Enhance top-of-funnel processes by improving lead scoring and qualification criteria. This will ensure that only high-potential leads progress through the funnel, leading to a higher conversion rate downstream.
         
• Monitor and Adapt:
    Continuously track key performance indicators and conversion metrics. Regularly review campaign performance and operational processes to make data-driven adjustments that optimize the overall efficiency of your customer acquisition strategy.
""")