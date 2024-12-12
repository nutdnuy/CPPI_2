import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GBM
import io
import plotly.express as px

st.set_page_config(page_title="CPPI Monte Carlo Simulation", layout="wide")

def show_cppi(n_scenarios=500, mu=0.1, sigma=0.15, m=3, floor=0.7, 
              riskfree_rate=0.05, steps_per_year=12, y_max=100):
    start = 10
    sim_rets = GBM.gbm(
        n_scenarios=n_scenarios, 
        mu=mu, 
        sigma=sigma, 
        prices=False, 
        steps_per_year=steps_per_year
    )
    risky_r = pd.DataFrame(sim_rets)
    # run the back-test
    btr = GBM.run_cppi(
        risky_r=pd.DataFrame(risky_r), 
        riskfree_rate=riskfree_rate, 
        m=m, 
        start=start, 
        floor=floor
    )
    wealth = btr["Wealth"]

    # calculate terminal wealth stats
    adj_y_max = wealth.values.max()*y_max/100 if y_max > 0 else wealth.values.max()
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios if n_scenarios > 0 else np.nan
    e_shortfall = np.dot(terminal_wealth - start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    # Plot with matplotlib
    fig, (wealth_ax, hist_ax) = plt.subplots(
        nrows=1, ncols=2, 
        sharey=True, 
        gridspec_kw={'width_ratios':[3,2]}, 
        figsize=(24, 9)
    )
    plt.subplots_adjust(wspace=0.0)

    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=adj_y_max)
    wealth_ax.set_title("Simulated Wealth Over Time", fontsize=16)

    terminal_wealth.plot.hist(
        ax=hist_ax, 
        bins=50, 
        ec='w', 
        fc='indianred', 
        orientation='horizontal'
    )
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=16)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=16)

    if floor > 0.01:
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        # Add the annotation for violations and expected shortfall
        hist_ax.annotate(
            f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", 
            xy=(.7, .7), 
            xycoords='axes fraction', 
            fontsize=24
        )


    

    hist_ax.set_title("Distribution of Terminal Wealth", fontsize=16)

    return fig, p_fail, e_shortfall, wealth

st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 3em; margin-bottom: 0;">CPPI Monte Carlo Simulation</h1>
    <h3 style="color: gray; margin-top: 0;">Exploring Cushion Portfolio Protection Strategies</h3>
</div>
<hr>
""", unsafe_allow_html=True)

st.write("""
**What is CPPI?**  
Constant Proportion Portfolio Insurance (CPPI) is a strategy that allocates between a risky asset and a risk-free asset to limit downside risk while trying to capture upside returns.

Use the controls below to set the parameters of the simulation. Click **Submit** to run the simulation and explore the results.
""")

with st.expander("Learn more about CPPI and this simulation"):
    st.write("""
    **How the Strategy Works**:
    - You start with a certain initial wealth (e.g., $10).
    - A "floor" level is set as a fraction of your initial wealth.
    - The "cushion" is (current portfolio value - floor value).
    - The strategy invests a multiple (m) of the cushion in the risky asset, with the rest in the risk-free asset.
    
    **What the Simulation Does**:
    - Runs a Monte Carlo simulation of risky asset returns based on mu (expected return) and sigma (volatility).
    - Applies the CPPI strategy rules at specified intervals.
    - Analyzes the final distribution of terminal wealth to understand performance and downside risks.
    """)



with st.expander("Limitation for montecalo simulation"):
    st.write("""
    **1. Assumptions about Market Behavior**:
    - Geometric Brownian Motion (GBM).
    - Stationarity: Simulations assume that market parameters like expected return and volatility remain constant over time, which is unrealistic for long-term horizons.
    - No Extreme Events
    
    **2. Over-Simplification of Portfolio Dynamics**:
    - Transaction Costs: Frequent rebalancing in CPPI generates transaction costs, which can significantly reduce portfolio performance. These are often not included in simulations.
    - Liquidity Constraints: In real markets, there may be limitations on how quickly or efficiently an investor can rebalance a portfolio, especially during volatile periods.
    - Borrowing Constraints: The strategy assumes perfect access to capital markets to rebalance allocations, which may not be possible during extreme market stress.
    """)



with st.form("cppi_form"):
    st.markdown("### Simulation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        n_scenarios = st.slider("Number of Scenarios", 1, 2000, 500, 5)
        mu = st.slider("Expected Return (mu)", 0.0, 0.2, 0.1, 0.01)
        sigma = st.slider("Volatility (sigma)", 0.0, 0.3, 0.15, 0.05)
        floor = st.slider("Floor (fraction of initial)", 0.0, 2.0, 0.7, 0.1)

    with col2:
        m = st.slider("Multiplier (m)", 1.0, 5.0, 3.0, 0.5)
        riskfree_rate = st.slider("Risk-free rate", 0.0, 0.05, 0.05, 0.01)
        steps_per_year = st.slider("Rebalancings per Year", 1, 12, 12, 1)
        y_max = st.slider("Zoom Y Axis (%)", 0, 100, 100, 1)

    submitted = st.form_submit_button("Submit")

if submitted:
    start = 10
    with st.spinner("Running simulation..."):
        fig, p_fail, e_shortfall, wealth = show_cppi(
            n_scenarios=n_scenarios,
            mu=mu,
            sigma=sigma,
            m=m,
            floor=floor,
            riskfree_rate=riskfree_rate,
            steps_per_year=steps_per_year,
            y_max=y_max
        )

    st.pyplot(fig)
    
    # Display Results
    st.markdown("### Results")
    col1, col2 = st.columns(2)
    col1.metric("Probability of Failure", f"{p_fail:.2%}")
    col2.metric("Expected Shortfall (if failure)", f"${e_shortfall:,.2f}")

    if p_fail > 0.2:
        st.warning("High probability of failure suggests aggressive parameters or harsh conditions.")
    elif 0 < p_fail <= 0.2:
        st.info("Moderate probability of failure. Consider adjusting parameters.")
    else:
        st.success("No failures recorded!")

    # Download Wealth Data
    csv_buffer = io.StringIO()
    wealth.to_csv(csv_buffer)
    st.download_button(
        label="Download Wealth Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="wealth_data.csv",
        mime="text/csv"
    )

    # Analyze Terminal Wealth
    terminal_wealth = wealth.iloc[-1]

    # --- 1) Percentile Table ---
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(terminal_wealth, percentiles)

    df_percentiles = pd.DataFrame({
        "Percentile": [f"{p}th" for p in percentiles],
        "Terminal Wealth": [f"${v:,.2f}" for v in percentile_values]
    })

    # --- 2) Probability of Surpassing Certain Wealth Thresholds ---
    # Let's pick some example thresholds:
    thresholds = [start, start*1.5, start*2, start*3, start*5]
    probs = [(terminal_wealth >= t).mean() for t in thresholds]

    df_thresholds = pd.DataFrame({
        "Wealth Threshold": [f"${t:,.2f}" for t in thresholds],
        "Probability (≥ Threshold)": [f"{p*100:.2f}%" for p in probs]
    })

    # --- 3) Loss Probability Table ---
    # Define a set of loss thresholds (as percentages of initial wealth).
    # A loss threshold of 10% means final wealth < start*0.9.
    loss_thresholds = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # Calculate probability that loss >= each threshold
    # Loss >= X% means: terminal_wealth < start*(1 - X)
    loss_probs = []
    for lt in loss_thresholds:
        cutoff = start*(1 - lt)
        loss_prob = (terminal_wealth < cutoff).mean()
        loss_probs.append(loss_prob)

    df_loss_prob = pd.DataFrame({
        "Loss Threshold": [f">= {lt*100:.2f}%" for lt in loss_thresholds],
        "Probability of Loss": [f"{lp*100:.2f}%" for lp in loss_probs]
    })


    # Plot Terminal Wealth Histogram (bottom 95% of results)
    terminal_wealth = wealth.iloc[-1]
    cutoff_tw = terminal_wealth.quantile(0.95)
    filtered_tw = terminal_wealth[terminal_wealth <= cutoff_tw]
    
    df_filtered_tw = filtered_tw.to_frame(name="terminal_wealth_value").reset_index(drop=True)
    
    fig_hist_tw = px.histogram(
        df_filtered_tw,
        x="terminal_wealth_value",
        nbins=50,
        title='Portfolio End Balance Histogram (95% of results)',
        template="simple_white"
    )
    
    fig_hist_tw.update_traces(marker_color='rgba(0, 102, 204, 0.8)', 
                              marker_line_color='rgba(0, 102, 204, 1.0)', 
                              marker_line_width=1)
    
    fig_hist_tw.update_layout(
        bargap=0.1,
        title_font_size=24,
        title_font_color='navy',
        xaxis_title='End Balance ($)',
        yaxis_title='Frequency',
        title_x=0.5,
        font=dict(family="Arial", size=14, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig_hist_tw.update_xaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor='navy', 
        mirror=True, 
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )
    fig_hist_tw.update_yaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor='navy', 
        mirror=True, 
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )

    st.plotly_chart(fig_hist_tw, use_container_width=True)

    # Compute and plot Maximum Drawdown Histogram
    # Max Drawdown per scenario: (Wealth/Wealth.cummax() - 1).min() along the time index
    drawdowns = wealth / wealth.cummax() - 1
    max_drawdowns = drawdowns.min()  # one per column/scenario
    
    cutoff_mdd = max_drawdowns.quantile(0.95)
    filtered_mdd = max_drawdowns[max_drawdowns <= cutoff_mdd]
    df_mdd = filtered_mdd.to_frame(name="max_drawdown").reset_index(drop=True)

    fig_mdd = px.histogram(
        df_mdd,
        x="max_drawdown",
        nbins=50,
        title='Maximum Drawdown Histogram (95% of results)',
        template="simple_white"
    )

    # Let's choose a nice orange color
    fig_mdd.update_traces(marker_color='rgba(255,140,0,0.9)', 
                          marker_line_color='white', 
                          marker_line_width=1)

    fig_mdd.update_layout(
        bargap=0.1,
        title_font_size=24,
        title_font_color='blue',
        xaxis_title='Max. Drawdown',
        yaxis_title='Frequency',
        title_x=0.5,
        font=dict(family="Arial", size=14, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Format x-axis as percentage
    fig_mdd.update_xaxes(
        tickformat=".1%",  
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
        mirror=True, 
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )
    fig_mdd.update_yaxes(
        showline=True, 
        linewidth=1.5, 
        linecolor='black', 
        mirror=True, 
        gridcolor='lightgray',
        tickfont=dict(color='black')
    )

    st.plotly_chart(fig_mdd, use_container_width=True)


    st.markdown("### Terminal Wealth Percentiles")
    st.table(df_percentiles.style.set_properties(**{'text-align': 'center'}))

    st.markdown("### Probability of Surpassing Wealth Thresholds")
    st.table(df_thresholds.style.set_properties(**{'text-align': 'center'}))

    st.markdown("### Loss Probabilities")
    st.table(df_loss_prob.style.set_properties(**{'text-align': 'center'}))

else:
    st.info("Adjust the parameters and click Submit to run the simulation.")

