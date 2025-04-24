import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(page_title="SKU Price Simulator", layout="centered")
st.title("📦 Toothpaste SKU Price & Size Simulator")

st.markdown("""
Use this tool to simulate the profitability of toothpaste SKUs based on:
- Specific fixed costs for each size (in EGP)
- Exchange rate to SAR
- Target retail price in the Saudi market
- Distributor and retailer margin assumptions
- Investment constraints in EGP
""")

# --- User Inputs Section ---
st.sidebar.header("🔧 Input Parameters")
exchange_rate = st.sidebar.number_input("EGP to SAR Exchange Rate", value=0.12, step=0.005, format="%.3f")

# Initialize default size-cost pairs
if 'size_costs' not in st.session_state:
    st.session_state.size_costs = {
        "50ml": 13.0,
        "75ml": 17.0,
        "100ml": 22.0
    }

# Size and cost section
st.sidebar.subheader("Product Sizes and Costs")

# Table-like UI for size-cost pairs
st.sidebar.markdown("##### Size-Cost Configuration")

# Add new size-cost pair button
if st.sidebar.button("Add New Size"):
    # Add a new empty size-cost pair
    new_key = f"{len(st.session_state.size_costs) + 1}0ml"
    st.session_state.size_costs[new_key] = 15.0

# Display and manage all size-cost pairs
sizes_to_remove = []
updated_size_costs = {}

for size_name, cost in st.session_state.size_costs.items():
    col1, col2, col3, col4 = st.sidebar.columns([0.3, 0.25, 0.35, 0.1])
    
    # Extract numeric value from size name (e.g., "50ml" -> 50)
    try:
        current_size = int(''.join(filter(str.isdigit, size_name)))
    except:
        current_size = 50
    
    # Size input (ml)
    new_size = col1.number_input(
        "Size (ml)", 
        value=current_size, 
        step=5, 
        key=f"size_value_{size_name}"
    )
    
    # Create new size name
    new_size_name = f"{new_size}ml"
    
    # Cost input (EGP)
    new_cost = col2.number_input(
        "Cost (EGP)", 
        value=float(cost), 
        step=0.5, 
        format="%.1f", 
        key=f"cost_{size_name}"
    )
    
    # Calculate and display cost per ml
    cost_per_ml = round(new_cost / new_size, 2)
    col3.metric("EGP/ml", f"{cost_per_ml}")
    
    # Remove button
    if len(st.session_state.size_costs) > 1 and col4.button("❌", key=f"remove_{size_name}"):
        sizes_to_remove.append(size_name)
    else:
        updated_size_costs[new_size_name] = new_cost

# Remove marked sizes
for size_name in sizes_to_remove:
    if size_name in st.session_state.size_costs:
        del st.session_state.size_costs[size_name]

# Update with any renamed sizes
st.session_state.size_costs = updated_size_costs

# Convert EGP costs to SAR
sizes_costs_sar = {size: round(cost * exchange_rate, 2) for size, cost in st.session_state.size_costs.items()}

# Display converted costs
st.sidebar.subheader("Converted Costs (SAR)")
for size, cost in sizes_costs_sar.items():
    st.sidebar.markdown(f"**{size}:** {cost:.2f} SAR")

# Pricing inputs
distributor_margin_pct = st.sidebar.slider("Distributor Margin %", 5, 40, 25, step=1)
retail_margin_pct = st.sidebar.slider("Retail Margin %", 5, 30, 15, step=1)

# Generate price options based on costs
min_price = min(sizes_costs_sar.values()) * 1.5  # Minimum viable price
max_price = max(sizes_costs_sar.values()) * 2.5  # Maximum reasonable price
price_range = [round(p, 1) for p in [*range(int(min_price*10), int(max_price*10)+1, 5)]]
price_range = [p/10 for p in price_range]  # Convert back to float

target_prices = st.sidebar.multiselect(
    "Select Target Retail Prices (SAR)",
    options=price_range,
    default=[price_range[len(price_range)//3], price_range[2*len(price_range)//3]]
)

revenue_goal = st.sidebar.number_input("Target Revenue Goal (SAR)", value=10000, step=500)
investment_cap_egp = st.sidebar.number_input("Maximum Investment (EGP)", value=250000, step=10000)

# --- Calculation Section ---
results = []
for size_name, cost_egp in st.session_state.size_costs.items():
    size_ml = int(''.join(filter(str.isdigit, size_name)))
    cost_sar = sizes_costs_sar[size_name]
    
    for price in target_prices:
        # Calculate margins
        distributor_margin = round(cost_sar * distributor_margin_pct / 100, 2)
        retail_margin = round(cost_sar * retail_margin_pct / 100, 2)
        total_chain_cost = cost_sar + distributor_margin + retail_margin
        remaining = round(price - total_chain_cost, 2)
        margin_percentage = round((remaining / price) * 100, 1)

        # Calculate business metrics
        units_needed = round(revenue_goal / price)
        total_profit = round(units_needed * remaining, 2)
        total_investment_egp = round(units_needed * cost_egp, 2)
        roi_percentage = round((total_profit / (total_investment_egp * exchange_rate)) * 100, 1)

        # Dynamic recommendation
        if total_investment_egp > investment_cap_egp:
            recommendation = "❌ Exceeds Investment Cap"
        elif margin_percentage >= 8 and margin_percentage <= 15:
            recommendation = "✅ Optimal Margin"
        elif margin_percentage > 15:
            recommendation = "✅ High Margin – Consider Growth Opportunity"
        elif margin_percentage > 0 and margin_percentage < 8:
            recommendation = "⚠️ Low Margin – Recheck Pricing"
        else:
            recommendation = "❌ Not Profitable"

        results.append({
            "Size": size_name,
            "Size (ml)": size_ml,
            "Cost (EGP)": cost_egp,
            "Cost per ml (EGP)": round(cost_egp / size_ml, 2),
            "Cost (SAR)": cost_sar,
            "Target Price (SAR)": price,
            "Price per ml (SAR)": round(price / size_ml, 2),
            "Distributor Margin": distributor_margin,
            "Retail Margin": retail_margin,
            "Total Chain Cost": total_chain_cost,
            "Remaining Margin": remaining,
            "Margin %": margin_percentage,
            f"Units for {revenue_goal} SAR Revenue": units_needed,
            f"Total Profit at {revenue_goal} SAR": total_profit,
            "Total Investment (EGP)": total_investment_egp,
            "ROI %": roi_percentage,
            "Recommendation": recommendation
        })

results_df = pd.DataFrame(results)

# --- Filter and Display Results ---
st.subheader("📊 SKU Simulation Results")

# Filters for interactive exploration
col1, col2 = st.columns(2)
size_filter = col1.multiselect(
    "Filter by Size",
    options=results_df["Size"].unique(),
    default=results_df["Size"].unique()
)

price_filter = col2.multiselect(
    "Filter by Target Price (SAR)",
    options=results_df["Target Price (SAR)"].unique(),
    default=results_df["Target Price (SAR)"].unique()
)

# Apply filters
filtered_df = results_df[
    results_df["Size"].isin(size_filter) & 
    results_df["Target Price (SAR)"].isin(price_filter)
]

# Sort results by profitability
filtered_df = filtered_df.sort_values(by=["Recommendation", "Margin %", "ROI %"], ascending=[True, False, False])

# Define formatting per column
format_dict = {
    "Size (ml)": "{:.0f}",
    "Cost (EGP)": "{:.1f}",
    "Cost per ml (EGP)": "{:.2f}",
    "Cost (SAR)": "{:.2f}",
    "Target Price (SAR)": "{:.1f}",
    "Price per ml (SAR)": "{:.3f}",
    "Distributor Margin": "{:.2f}",
    "Retail Margin": "{:.2f}",
    "Total Chain Cost": "{:.2f}",
    "Remaining Margin": "{:.2f}",
    "Margin %": "{:.1f}%",
    f"Units for {revenue_goal} SAR Revenue": "{:.0f}",
    f"Total Profit at {revenue_goal} SAR": "{:.1f}",
    "Total Investment (EGP)": "{:.0f}",
    "ROI %": "{:.1f}%"
}

st.dataframe(filtered_df.style.format(format_dict), use_container_width=True)

# Show recommendation summary
profitable = filtered_df[filtered_df["Recommendation"].str.contains("✅")]
if not profitable.empty:
    st.success("✅ Recommended SKUs with feasible margin and investment")
    st.dataframe(profitable[["Size", "Target Price (SAR)", "Margin %", "ROI %", "Total Investment (EGP)", "Recommendation"]].style.format(format_dict), use_container_width=True)
else:
    st.warning("⚠️ No SKU fits margin or investment constraints. Try adjusting inputs.")

# --- Visualization Section ---

# Determine optimal combinations for highlighting
optimal_combinations = filtered_df[filtered_df["Recommendation"].str.contains("✅")].copy()
if optimal_combinations.empty:
    optimal_indicator = filtered_df["Recommendation"]
else:
    # Create an indicator column for optimal combinations
    filtered_df["Is Optimal"] = filtered_df.apply(
        lambda row: "Optimal Choice" if row["Recommendation"].startswith("✅") else "Other Options", 
        axis=1
    )

# Visual 1: Margin % Comparison
fig1 = px.bar(
    filtered_df,
    x="Size",
    y="Margin %",
    color="Target Price (SAR)",
    barmode="group",
    title="Profit Margin % by Size and Target Price",
    color_continuous_scale="Viridis"
)
fig1.add_hline(y=8, line_dash="dash", line_color="green", 
              annotation_text="Minimum Target Margin (8%)")
st.plotly_chart(fig1, use_container_width=True)

# Visual 2: Price vs. Margin Analysis
fig2 = px.scatter(
    filtered_df,
    x="Target Price (SAR)",
    y="Margin %",
    color="Size",
    size="Size (ml)", 
    hover_data=["ROI %", "Total Investment (EGP)", "Recommendation"],
    title="Price vs. Margin Analysis"
)
fig2.add_hline(y=8, line_dash="dash", line_color="green")
fig2.add_hline(y=15, line_dash="dash", line_color="orange")
st.plotly_chart(fig2, use_container_width=True)

# Visual 3: ROI Comparison
fig3 = px.bar(
    filtered_df,
    x="Size",
    y="ROI %",
    color="Target Price (SAR)",
    barmode="group",
    title="Return on Investment (%) by Size and Price"
)
st.plotly_chart(fig3, use_container_width=True)

# Visual 4: Investment vs. Profit
fig4 = px.scatter(
    filtered_df,
    x="Total Investment (EGP)",
    y=f"Total Profit at {revenue_goal} SAR",
    color="Size",
    symbol="Recommendation",
    size="Target Price (SAR)",
    hover_data=["Margin %", "ROI %"],
    title=f"Total Profit vs Investment (Cap = {investment_cap_egp:,.0f} EGP)"
)
fig4.add_vline(x=investment_cap_egp, line_dash="dash", line_color="red")
st.plotly_chart(fig4, use_container_width=True)

# Visual 5: Cost Structure Breakdown
selected_size = st.selectbox("Select Size for Cost Structure Breakdown", options=filtered_df["Size"].unique(), index=0)
selected_price = st.selectbox("Select Price for Cost Structure Breakdown", options=filtered_df["Target Price (SAR)"].unique(), index=0)

# Filter data for selected configuration
selected_config = filtered_df[
    (filtered_df["Size"] == selected_size) & 
    (filtered_df["Target Price (SAR)"] == selected_price)
]

if not selected_config.empty:
    row = selected_config.iloc[0]
    
    # Prepare data for waterfall chart
    cost_components = [
        {"name": "Cost (SAR)", "value": row["Cost (SAR)"]},
        {"name": "Distributor Margin", "value": row["Distributor Margin"]},
        {"name": "Retail Margin", "value": row["Retail Margin"]},
        {"name": "Remaining Margin", "value": row["Remaining Margin"]}
    ]
    
    # Custom waterfall chart for cost structure
    fig5 = go.Figure()
    
    # Add bars for each component
    colors = ["lightblue", "lightgreen", "lightyellow", 
              "lightpink" if row["Remaining Margin"] > 0 else "red"]
    
    cumulative = 0
    for i, component in enumerate(cost_components):
        fig5.add_trace(go.Bar(
            name=component["name"],
            y=[component["name"]],
            x=[component["value"]],
            orientation="h",
            marker=dict(color=colors[i]),
            text=f"{component['value']:.2f} SAR",
            textposition="auto"
        ))
        cumulative += component["value"]
    
    # Add total line
    fig5.add_trace(go.Scatter(
        name="Total",
        y=["Total"],
        x=[row["Target Price (SAR)"]],
        mode="markers+text",
        marker=dict(size=15, color="black"),
        text=f"{row['Target Price (SAR)']:.2f} SAR",
        textposition="middle right"
    ))
    
    fig5.update_layout(
        title=f"Cost Structure for {selected_size} at {selected_price} SAR",
        barmode="stack",
        xaxis_title="SAR",
        showlegend=False
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Display key metrics for this configuration
    col1, col2, col3 = st.columns(3)
    col1.metric("Margin %", f"{row['Margin %']:.1f}%")
    col2.metric("ROI %", f"{row['ROI %']:.1f}%")
    col3.metric("Units Needed", f"{row[f'Units for {revenue_goal} SAR Revenue']:,.0f}")

# Visual 6: Investment Distribution
fig6 = px.pie(
    filtered_df,
    values="Total Investment (EGP)",
    names="Size",
    title="Investment Distribution by Size",
    hole=0.4
)
st.plotly_chart(fig6, use_container_width=True)

# Add a download button for the results
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="toothpaste_sku_simulation_results.csv",
    mime="text/csv",
)

# Final insights and recommendations
st.subheader("🔎 Key Insights")

if profitable.empty:
    st.warning("No profitable SKU combinations found with current parameters.")
else:
    # Get the most profitable SKU by margin
    best_margin = profitable.loc[profitable["Margin %"].idxmax()]
    
    # Get the best ROI SKU
    best_roi = profitable.loc[profitable["ROI %"].idxmax()]
    
    # Get the most efficient investment (lowest investment with acceptable margin)
    efficiency_factor = profitable["Margin %"] / profitable["Total Investment (EGP)"] * 1000
    most_efficient = profitable.loc[efficiency_factor.idxmax()]
    
    st.markdown(f"""
    Based on the analysis, here are the top recommendations:
    
    1. **Best Profit Margin**: {best_margin['Size']} at {best_margin['Target Price (SAR)']} SAR 
       - Margin: {best_margin['Margin %']:.1f}%
       - Investment: {best_margin['Total Investment (EGP)']:,.0f} EGP
       
    2. **Best ROI**: {best_roi['Size']} at {best_roi['Target Price (SAR)']} SAR
       - ROI: {best_roi['ROI %']:.1f}%
       - Investment: {best_roi['Total Investment (EGP)']:,.0f} EGP
       
    3. **Most Efficient Investment**: {most_efficient['Size']} at {most_efficient['Target Price (SAR)']} SAR
       - Requires only {most_efficient['Total Investment (EGP)']:,.0f} EGP investment
       - Provides {most_efficient['Margin %']:.1f}% margin
    """)
    
    # Add specific market fit advice
    if len(profitable["Size"].unique()) > 1:
        st.markdown("""
        **Market Fit Strategy:**
        - Consider launching multiple sizes to target different consumer segments
        - Higher-priced, larger sizes may attract value-conscious consumers
        - Smaller sizes can serve as entry points for new customers
        """)
