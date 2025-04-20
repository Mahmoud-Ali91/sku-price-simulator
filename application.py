import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(page_title="SKU Price Simulator", layout="centered")
st.title("📦 Toothpaste SKU Price & Size Simulator")

st.markdown("""
Use this tool to simulate the profitability of fixed toothpaste sizes based on:
- Cost per ml in EGP
- Exchange rate to SAR
- Converted cost per ml (SAR) for specific packaging
- Target retail price in the Saudi market
- Distributor and retailer margin assumptions
- Investment constraints in EGP
""")

# --- User Inputs Section ---
st.sidebar.header("🔧 Input Parameters")
exchange_rate = st.sidebar.number_input("EGP to SAR Exchange Rate", value=0.12, step=0.005)
cost_per_ml_egp = st.sidebar.number_input("Estimated Cost per ml (EGP)", value=1.80, step=0.05)
cost_per_ml_sar = round(cost_per_ml_egp * exchange_rate, 3)
st.sidebar.markdown(f"**Converted Cost per ml (SAR):** {cost_per_ml_sar:.3f}")


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
        factory_cost = round(size * cost_per_ml_sar, 2)
        distributor_margin = round(factory_cost * distributor_margin_pct / 100, 2)
        retail_margin = round(factory_cost * retail_margin_pct / 100, 2)
        total_chain_cost = factory_cost + distributor_margin + retail_margin
        remaining = round(price - total_chain_cost, 2)

        units_needed = round(revenue_goal / price)
        total_profit = round(units_needed * remaining, 2)
        total_investment_egp = round(units_needed * size * cost_per_ml_egp, 2)

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
            "Target Price (SAR)": price,
            "Fixed Size (ml)": size,
            "Cost per ml (EGP)": round(cost_per_ml_egp, 2),
            "Exchange Rate": exchange_rate,
            "Cost per ml (SAR)": round(cost_per_ml_sar, 3),
            "Factory Cost": factory_cost,
            "Distributor Margin": distributor_margin,
            "Retail Margin": retail_margin,
            "Total Chain Cost": total_chain_cost,
            "Remaining Margin": remaining,
            f"Total Units Needed to Hit {revenue_goal} SAR Revenue": units_needed,
            "Profit per Unit (SAR)": remaining,
            f"Total Profit at {revenue_goal} SAR Revenue": total_profit,
            "Total Investment (EGP)": total_investment_egp,
            "Recommendation": recommendation
        })

results_df = pd.DataFrame(results)

# --- Display Results ---
st.subheader("📊 SKU Simulation Results")

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
    f"Total Units Needed to Hit {revenue_goal} SAR Revenue": "{:.0f}",
    "Profit per Unit (SAR)": "{:.2f}",
    f"Total Profit at {revenue_goal} SAR Revenue": "{:.2f}",
    "Total Investment (EGP)": "{:.0f}"
}

st.dataframe(results_df.style.format(format_dict), use_container_width=True)

# Show recommendation summary
profitable = results_df[results_df["Recommendation"].str.contains("✅")]
if not profitable.empty:
    st.success("✅ Recommended SKUs with feasible margin and price")
    st.dataframe(profitable[["Target Price (SAR)", "Fixed Size (ml)", "Recommendation"]])
else:
    st.warning("⚠️ No SKU fits margin or investment constraints. Try adjusting inputs.")
...

# --- Visualization Section ---

# Visual 1: Remaining Margin per Size & Price
fig1 = px.bar(
    results_df,
    x="Fixed Size (ml)",
    y="Remaining Margin",
    color="Target Price (SAR)",
    barmode="group",
    title="Remaining Margin per Fixed Size & Price"
)
st.plotly_chart(fig1, use_container_width=True)

# Visual 2: Profit per Unit Distribution
fig2 = px.scatter(
    results_df,
    x="Target Price (SAR)",
    y="Profit per Unit (SAR)",
    color="Recommendation",
    size="Fixed Size (ml)",
    title="Profit per Unit vs Retail Price",
    hover_data=["Fixed Size (ml)", "Total Investment (EGP)"]
)
st.plotly_chart(fig2, use_container_width=True)

# Visual 3: Total Investment per SKU
fig3 = px.bar(
    results_df,
    x="Fixed Size (ml)",
    y="Total Investment (EGP)",
    color="Target Price (SAR)",
    barmode="group",
    title="Total Investment in EGP per SKU Option"
)
st.plotly_chart(fig3, use_container_width=True)

# Visual 4: Total Profit vs Investment Cap
fig4 = px.scatter(
    results_df,
    x="Total Investment (EGP)",
    y=f"Total Profit at {revenue_goal} SAR Revenue",
    color="Recommendation",
    symbol="Fixed Size (ml)",
    title=f"Total Profit vs Investment (Cap = {investment_cap_egp} EGP)"
)
st.plotly_chart(fig4, use_container_width=True)
# Break-even point: how many units needed to recover investment at this profit/unit
if remaining > 0:
            break_even_units = round(investment_cap_egp / (remaining * exchange_rate), 0)
    else:
            break_even_units = float('inf')  # not feasible

        results.append({
            "Target Price (SAR)": price,
            "Fixed Size (ml)": size,
            "Dynamic Cost per ml (EGP)": round(dynamic_cost_egp, 2),
            "Exchange Rate": exchange_rate,
            "Cost per ml (SAR)": round(dynamic_cost_sar, 3),
            "Factory Cost": factory_cost,
            "Distributor Margin": distributor_margin,
            "Retail Margin": retail_margin,
            "Total Chain Cost": total_chain_cost,
            "Remaining Margin": remaining,
            f"Total Units Needed to Hit {revenue_goal} SAR Revenue": units_needed,
            "Profit per Unit (SAR)": remaining,
            f"Total Profit at {revenue_goal} SAR Revenue": total_profit,
            "Total Investment (EGP)": total_investment_egp,
            "Break-even Units (based on Investment Cap)": break_even_units,
            "Recommendation": recommendation
        })
