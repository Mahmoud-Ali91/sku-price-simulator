import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(page_title="SKU Price Simulator", layout="centered")
st.title("📦 Toothpaste SKU Price & Size Simulator")

st.markdown("""
Use this tool to simulate the profitability of fixed toothpaste sizes based on:
- Multiple cost scenarios per ml in EGP
- Exchange rate to SAR
- Converted cost per ml (SAR) for specific packaging
- Target retail price in the Saudi market
- Distributor and retailer margin assumptions
- Investment constraints in EGP
""")

# --- User Inputs Section ---
st.sidebar.header("🔧 Input Parameters")
exchange_rate = st.sidebar.number_input("EGP to SAR Exchange Rate", value=0.12, step=0.005, format="%.3f")

# Multiple costs per ml section
st.sidebar.subheader("Cost per ml Scenarios (EGP)")
# Default costs for initialization
default_costs = {"Standard": 1.80, "Premium": 2.10, "Economy": 1.50}

# Get the number of cost scenarios from session state or initialize
if 'cost_scenarios' not in st.session_state:
    st.session_state.cost_scenarios = default_costs

# Add new cost scenario button
if st.sidebar.button("Add New Cost Scenario"):
    # Add a new empty cost scenario
    new_key = f"Scenario {len(st.session_state.cost_scenarios) + 1}"
    st.session_state.cost_scenarios[new_key] = 1.75

# Display all cost scenarios with option to remove
costs_to_remove = []
updated_costs = {}

for name, cost in st.session_state.cost_scenarios.items():
    col1, col2, col3 = st.sidebar.columns([0.45, 0.45, 0.1])
    
    # Text input for scenario name
    new_name = col1.text_input(f"Name", value=name, key=f"name_{name}")
    
    # Number input for cost value
    new_cost = col2.number_input(
        f"Cost (EGP)", 
        value=float(cost), 
        step=0.05, 
        format="%.2f", 
        key=f"cost_{name}"
    )
    
    # Remove button
    if len(st.session_state.cost_scenarios) > 1 and col3.button("❌", key=f"remove_{name}"):
        costs_to_remove.append(name)
    else:
        updated_costs[new_name] = new_cost

# Remove marked scenarios
for name in costs_to_remove:
    if name in st.session_state.cost_scenarios:
        del st.session_state.cost_scenarios[name]

# Update with any renamed scenarios
st.session_state.cost_scenarios = updated_costs

# Convert EGP costs to SAR
costs_per_ml_sar = {name: round(cost * exchange_rate, 3) for name, cost in st.session_state.cost_scenarios.items()}

# Display converted costs
st.sidebar.subheader("Converted Costs (SAR)")
for name, cost in costs_per_ml_sar.items():
    st.sidebar.markdown(f"**{name}:** {cost:.3f} SAR/ml")

# Pricing and size inputs
distributor_margin_pct = st.sidebar.slider("Distributor Margin %", 5, 40, 25, step=1)
retail_margin_pct = st.sidebar.slider("Retail Margin %", 5, 30, 15, step=1)
target_prices = st.sidebar.multiselect(
    "Select Target Retail Prices (SAR)",
    options=[6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
    default=[7.0]
)

sizes_ml = st.sidebar.multiselect(
    "Select Fixed Sizes (ml)",
    options=[50, 75, 100],
    default=[50, 75, 100]
)

revenue_goal = st.sidebar.number_input("Target Revenue Goal (SAR)", value=10000, step=500)
investment_cap_egp = st.sidebar.number_input("Maximum Investment (EGP)", value=250000, step=10000)

# --- Calculation Section ---
results = []
for size in sizes_ml:
    for price in target_prices:
        for cost_name, cost_egp in st.session_state.cost_scenarios.items():
            cost_sar = costs_per_ml_sar[cost_name]
            factory_cost = round(size * cost_sar, 2)
            distributor_margin = round(factory_cost * distributor_margin_pct / 100, 2)
            retail_margin = round(factory_cost * retail_margin_pct / 100, 2)
            total_chain_cost = factory_cost + distributor_margin + retail_margin
            remaining = round(price - total_chain_cost, 2)

            units_needed = round(revenue_goal / price)
            total_profit = round(units_needed * remaining, 2)
            total_investment_egp = round(units_needed * size * cost_egp, 2)

            # Dynamic recommendation
            if total_investment_egp > investment_cap_egp:
                recommendation = "❌ Exceeds Investment Cap"
            elif remaining >= 0.3 and remaining <= 0.8:
                recommendation = "✅ Profitable SKU"
            elif remaining > 0.8:
                recommendation = "✅ High Margin – Consider Lower Price or Bigger Size"
            elif remaining >= 0:
                recommendation = "⚠️ Low Margin – Recheck Costs or Margins"
            else:
                recommendation = "❌ Not Profitable"

            results.append({
                "Cost Scenario": cost_name,
                "Target Price (SAR)": price,
                "Fixed Size (ml)": size,
                "Cost per ml (EGP)": round(cost_egp, 2),
                "Exchange Rate": exchange_rate,
                "Cost per ml (SAR)": round(cost_sar, 3),
                "Factory Cost": factory_cost,
                "Distributor Margin": distributor_margin,
                "Retail Margin": retail_margin,
                "Total Chain Cost": total_chain_cost,
                "Remaining Margin": remaining,
                f"Units for {revenue_goal} SAR Revenue": units_needed,
                "Profit per Unit (SAR)": remaining,
                f"Total Profit at {revenue_goal} SAR": total_profit,
                "Total Investment (EGP)": total_investment_egp,
                "Recommendation": recommendation
            })

results_df = pd.DataFrame(results)

# --- Filter and Display Results ---
st.subheader("📊 SKU Simulation Results")

# Filters for interactive exploration
col1, col2 = st.columns(2)
cost_filter = col1.multiselect(
    "Filter by Cost Scenario", 
    options=results_df["Cost Scenario"].unique(),
    default=results_df["Cost Scenario"].unique()
)

size_filter = col2.multiselect(
    "Filter by Size (ml)",
    options=results_df["Fixed Size (ml)"].unique(),
    default=results_df["Fixed Size (ml)"].unique()
)

# Apply filters
filtered_df = results_df[
    results_df["Cost Scenario"].isin(cost_filter) & 
    results_df["Fixed Size (ml)"].isin(size_filter)
]

# Define formatting per column
format_dict = {
    "Target Price (SAR)": "{:.2f}",
    "Fixed Size (ml)": "{:.1f}",
    "Cost per ml (EGP)": "{:.2f}",
    "Exchange Rate": "{:.3f}",
    "Cost per ml (SAR)": "{:.3f}",
    "Factory Cost": "{:.2f}",
    "Distributor Margin": "{:.2f}",
    "Retail Margin": "{:.2f}",
    "Total Chain Cost": "{:.2f}",
    "Remaining Margin": "{:.2f}",
    f"Units for {revenue_goal} SAR Revenue": "{:.0f}",
    "Profit per Unit (SAR)": "{:.2f}",
    f"Total Profit at {revenue_goal} SAR": "{:.2f}",
    "Total Investment (EGP)": "{:.0f}"
}

st.dataframe(filtered_df.style.format(format_dict), use_container_width=True)

# Show recommendation summary
profitable = filtered_df[filtered_df["Recommendation"].str.contains("✅")]
if not profitable.empty:
    st.success("✅ Recommended SKUs with feasible margin and price")
    st.dataframe(profitable[["Cost Scenario", "Target Price (SAR)", "Fixed Size (ml)", "Recommendation"]])
else:
    st.warning("⚠️ No SKU fits margin or investment constraints. Try adjusting inputs.")

# --- Visualization Section ---

# Visual 1: Remaining Margin per Cost Scenario, Size & Price
fig1 = px.bar(
    filtered_df,
    x="Fixed Size (ml)",
    y="Remaining Margin",
    color="Cost Scenario",
    barmode="group",
    facet_col="Target Price (SAR)",
    title="Remaining Margin per Fixed Size & Cost Scenario"
)
st.plotly_chart(fig1, use_container_width=True)

# Visual 2: Profit per Unit Distribution
fig2 = px.scatter(
    filtered_df,
    x="Target Price (SAR)",
    y="Profit per Unit (SAR)",
    color="Cost Scenario",
    symbol="Recommendation", 
    size="Fixed Size (ml)",
    title="Profit per Unit vs Retail Price",
    hover_data=["Fixed Size (ml)", "Total Investment (EGP)"]
)
st.plotly_chart(fig2, use_container_width=True)

# Visual 3: Total Investment per SKU and Cost Scenario
fig3 = px.bar(
    filtered_df,
    x="Fixed Size (ml)",
    y="Total Investment (EGP)",
    color="Cost Scenario",
    pattern_shape="Target Price (SAR)",
    barmode="group",
    title="Total Investment in EGP per SKU Option"
)
fig3.add_hline(y=investment_cap_egp, line_dash="dash", line_color="red",
              annotation_text=f"Investment Cap: {investment_cap_egp:,.0f} EGP")
st.plotly_chart(fig3, use_container_width=True)

# Visual 4: Total Profit Comparison
fig4 = px.scatter(
    filtered_df,
    x="Total Investment (EGP)",
    y=f"Total Profit at {revenue_goal} SAR",
    color="Cost Scenario",
    symbol="Fixed Size (ml)",
    size="Target Price (SAR)",
    hover_data=["Recommendation"],
    title=f"Total Profit vs Investment (Cap = {investment_cap_egp:,.0f} EGP)"
)
fig4.add_vline(x=investment_cap_egp, line_dash="dash", line_color="red")
st.plotly_chart(fig4, use_container_width=True)

# Visual 5: Total Chain Cost Breakdown
subset_for_waterfall = filtered_df[filtered_df["Target Price (SAR)"] == filtered_df["Target Price (SAR)"].iloc[0]]
subset_for_waterfall = subset_for_waterfall.sort_values(by=["Cost Scenario", "Fixed Size (ml)"])

fig5 = go.Figure()

for i, row in subset_for_waterfall.iterrows():
    label = f"{row['Cost Scenario']} - {row['Fixed Size (ml)']}ml - {row['Target Price (SAR)]} SAR"
    
    fig5.add_trace(go.Bar(
        name=label,
        y=['Cost Structure'],
        x=[row['Factory Cost']],
        orientation='h',
        text="Factory",
        marker=dict(color='lightblue')
    ))
    
    fig5.add_trace(go.Bar(
        name=label,
        y=['Cost Structure'],
        x=[row['Distributor Margin']],
        orientation='h',
        text="Distributor",
        marker=dict(color='lightgreen')
    ))
    
    fig5.add_trace(go.Bar(
        name=label,
        y=['Cost Structure'],
        x=[row['Retail Margin']],
        orientation='h',
        text="Retail",
        marker=dict(color='lightyellow')
    ))
    
    fig5.add_trace(go.Bar(
        name=label,
        y=['Cost Structure'],
        x=[row['Remaining Margin']],
        orientation='h',
        text="Remaining",
        marker=dict(color='lightpink' if row['Remaining Margin'] > 0 else 'red')
    ))

fig5.update_layout(
    barmode='stack',
    title="Cost Structure Breakdown for Selected Price Point",
    xaxis_title="SAR",
    showlegend=False
)

st.plotly_chart(fig5, use_container_width=True)

# Add a download button for the results
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="toothpaste_sku_simulation_results.csv",
    mime="text/csv",
)
