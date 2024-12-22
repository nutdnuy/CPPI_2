import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GBM
import io
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="CPPI Monte Carlo Simulation", layout="wide")

def show_cppi(n_scenarios=500, mu=0.1, sigma=0.15, m=3, floor=0.7, 
              riskfree_rate=0.05, steps_per_year=12, y_max=100, start=10, n_years=10):
    """
    Run the CPPI simulation and return the figure, probability of failure, 
    expected shortfall, and wealth DataFrame.
    """
    sim_rets = GBM.gbm(
        n_scenarios=n_scenarios, 
        mu=mu, 
        sigma=sigma, 
        prices=False, 
        steps_per_year=steps_per_year,
        n_years=n_years
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

    # Plot wealth over time
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="#cc3300")  # using a distinct red shade
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=adj_y_max)
    wealth_ax.set_title("Simulated Wealth Over Time", fontsize=18)

    # Plot terminal wealth distribution
    terminal_wealth.plot.hist(
        ax=hist_ax, 
        bins=50, 
        ec='white', 
        fc='#66b3ff',  # using a blue shade
        orientation='horizontal'
    )
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":")
    hist_ax.axhline(y=tw_median, ls=":")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9), xycoords='axes fraction', fontsize=16)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85), xycoords='axes fraction', fontsize=16)

    if floor > 0.01:
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        # Add the annotation for violations and expected shortfall
        hist_ax.annotate(
            f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", 
            xy=(.7, .7), 
            xycoords='axes fraction', 
            fontsize=16,
            color='darkred'
        )

    hist_ax.set_title("Distribution of Terminal Wealth", fontsize=18)

    return fig, p_fail, e_shortfall, wealth

# Main Title Section
st.markdown("""
<div style="text-align: center; margin-top: -50px;">
    <h1 style="font-size: 3em; margin-bottom: 0;">CPPI Monte Carlo Simulation</h1>
    <h3 style="color: gray; margin-top: 0;">Explore Cushion Portfolio Protection Strategies</h3>
</div>
<hr style="border-top: 2px solid ">
""", unsafe_allow_html=True)

# Introductory Text
st.write("""
**Constant Proportion Portfolio Insurance (CPPI)** is a dynamic investment strategy designed to **limit downside risk** while still providing **upside growth potential**. The strategy continuously adjusts the proportion invested in a risky asset and a risk-free asset based on a "cushion" determined by a set floor.

Use the controls below to set the parameters of the simulation. Click **Submit** to run the simulation and explore the results.
""")

# Accordion-style explanations for more details
with st.expander("**What is CPPI?**", expanded=False):
    st.write("""
    **How the Strategy Works**:
    - You start with a certain initial wealth (e.g., $10).
    - A "floor" level is set as a fraction of your initial wealth.
    - The "cushion" = (current portfolio value - floor value).
    - The strategy invests a multiple (m) of the cushion in the risky asset and the rest in the risk-free asset.
    """)

with st.expander("**About This Simulation**", expanded=False):
    st.write("""
    **Monte Carlo Simulation**:
    - Runs multiple scenarios of risky asset returns using Geometric Brownian Motion (GBM).
    - Parameters (mu, sigma) define the expected return and volatility of the risky asset.
    - The CPPI rules are applied at each rebalancing step.

    **Interpreting the Results**:
    - Probability of Failure: The chance that your final wealth ends below the floor.
    - Expected Shortfall: The average shortfall amount given that you end below the floor.
    """)

with st.expander("**Limitations**", expanded=False):
    st.write("""
    **Assumptions**:
    - Market returns follow a GBM with constant mu and sigma.
    - No transaction costs or liquidity constraints.
    - Results are purely hypothetical and not indicative of real-world performance.
    """)

# Parameter Input Form
with st.form("cppi_form"):
    st.markdown("### Simulation Parameters")
    st.write("Adjust the inputs to define your simulation scenario.")

    # General Parameters
    st.markdown("**General Parameters**")
    col_general_1, col_general_2 = st.columns(2)
    with col_general_1:
        start = st.number_input(
            "Initial Start Wealth ($)", 
            min_value=1.0, 
            value=10.0, 
            step=1.0, 
            help="Your initial amount of investable wealth at the start of the simulation."
        )
    with col_general_2:
        n_years = st.select_slider(
            "Number of Years",
            options=range(1, 51), 
            value=10,
            help="How many years to run the simulation."
        )

    # Asset Parameters
    st.markdown("**Asset Parameters**")
    col_asset_1, col_asset_2 = st.columns(2)
    with col_asset_1:
        mu = st.slider(
            "Expected Return (mu)",
            0.0, 0.2, 0.1, 0.01,
            help="The annual expected return of the risky asset."
        )
        sigma = st.slider(
            "Volatility (sigma)",
            0.0, 0.3, 0.15, 0.05,
            help="The annual volatility of the risky asset."
        )
    with col_asset_2:
        riskfree_rate = st.slider(
            "Risk-free rate",
            0.0, 0.05, 0.05, 0.01,
            help="Annual risk-free return (e.g., from treasury bills)."
        )
        steps_per_year = st.slider(
            "Rebalancings per Year",
            1, 12, 12, 1,
            help="How many times per year the portfolio is rebalanced."
        )

    # CPPI Parameters
    st.markdown("**CPPI Strategy Parameters**")
    col_cppi_1, col_cppi_2 = st.columns(2)
    with col_cppi_1:
        n_scenarios = st.slider(
            "Number of Scenarios", 
            1, 2000000, 500, 5, 
            help="Number of simulation paths to run."
        )
        floor = st.slider(
            "Floor (fraction of initial)",
            0.0, 2.0, 0.7, 0.1,
            help="The floor as a fraction of your initial wealth. E.g., 0.7 means 70% of initial wealth."
        )
    with col_cppi_2:
        m = st.slider(
            "Multiplier (m)",
            1.0, 5.0, 3.0, 0.5,
            help="How aggressively to invest based on the cushion. Higher values = more aggressive."
        )
        y_max = st.slider(
            "Zoom Y Axis (%)",
            0, 100, 100, 1,
            help="Adjust the Y-axis to highlight the distribution of results."
        )

    # Submit Button with Icon
    submitted = st.form_submit_button(
        label="🚀 Run Simulation",
        help="Click to run the CPPI simulation with the selected parameters."
    )

if submitted:
    with st.spinner("Running simulation, please wait..."):
        fig, p_fail, e_shortfall, wealth = show_cppi(
            n_scenarios=n_scenarios,
            mu=mu,
            sigma=sigma,
            m=m,
            floor=floor,
            riskfree_rate=riskfree_rate,
            steps_per_year=steps_per_year,
            y_max=y_max,
            start=start,
            n_years=n_years
        )

    # Display the results as metrics
    st.markdown("### Key Results")
    col_res_1, col_res_2 = st.columns(2)
    col_res_1.metric("Probability of Failure", f"{p_fail:.2%}", help="The chance of ending below the set floor.")
    col_res_2.metric("Expected Shortfall", f"${e_shortfall:,.2f}", help="The average shortfall amount if failure occurs.")

    if p_fail > 0.2:
        st.warning("**High probability of failure.** Consider less aggressive parameters.")
    elif 0 < p_fail <= 0.2:
        st.info("**Moderate probability of failure.** You might want to adjust parameters.")
    else:
        st.success("**No failures recorded!** Your strategy appears resilient under these conditions.")

    # Show the plot
    st.pyplot(fig)

    # Download Wealth Data
    csv_buffer = io.StringIO()
    wealth.to_csv(csv_buffer)
    st.download_button(
        label="💾 Download Wealth Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="wealth_data.csv",
        mime="text/csv",
        help="Download the simulated wealth time series for all scenarios."
    )

    # Further Analysis
    terminal_wealth = wealth.iloc[-1]

    # Percentile Table
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(terminal_wealth, percentiles)
    df_percentiles = pd.DataFrame({
        "Percentile": [f"{p}th" for p in percentiles],
        "Terminal Wealth": [f"${v:,.2f}" for v in percentile_values]
    })

    # Probability of Surpassing Certain Wealth Thresholds
    thresholds = [start, start*1.5, start*2, start*3, start*5]
    probs = [(terminal_wealth >= t).mean() for t in thresholds]
    df_thresholds = pd.DataFrame({
        "Wealth Threshold": [f"${t:,.2f}" for t in thresholds],
        "Probability (≥ Threshold)": [f"{p*100:.2f}%" for p in probs]
    })

    # Loss Probability Table
    loss_thresholds = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    loss_probs = []
    for lt in loss_thresholds:
        cutoff = start*(1 - lt)
        loss_prob = (terminal_wealth < cutoff).mean()
        loss_probs.append(loss_prob)
    df_loss_prob = pd.DataFrame({
        "Loss Threshold": [f">= {lt*100:.2f}%" for lt in loss_thresholds],
        "Probability of Loss": [f"{lp*100:.2f}%" for lp in loss_probs]
    })

    # Additional Charts
    st.markdown("### Additional Distributions")

    # Terminal Wealth Histogram (Bottom 95%)
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
    fig_hist_tw.update_traces(marker_color='#0066cc', marker_line_color='white', marker_line_width=1)
    fig_hist_tw.update_layout(
        bargap=0.1,
        title_font_size=24,
        title_font_color='#4B0082',
        xaxis_title='End Balance ($)',
        yaxis_title='Frequency',
        title_x=0.5,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_hist_tw, use_container_width=True)

    # Maximum Drawdown Histogram
    drawdowns = wealth / wealth.cummax() - 1
    max_drawdowns = drawdowns.min()
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
    fig_mdd.update_traces(marker_color='#FF8C00', marker_line_color='white', marker_line_width=1)
    fig_mdd.update_layout(
        bargap=0.1,
        title_font_size=24,
        title_font_color='#4B0082',
        xaxis_title='Max. Drawdown',
        yaxis_title='Frequency',
        title_x=0.5,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_mdd.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_mdd, use_container_width=True)

    # Display Data Tables
    st.markdown("### Summary Tables")
    col_table_1, col_table_2, col_table_3 = st.columns(3)
    with col_table_1:
        st.markdown("**Terminal Wealth Percentiles**")
        st.table(df_percentiles.style.set_properties(**{'text-align': 'center'}))
    with col_table_2:
        st.markdown("**Probability of Surpassing Thresholds**")
        st.table(df_thresholds.style.set_properties(**{'text-align': 'center'}))
    with col_table_3:
        st.markdown("**Loss Probabilities**")
        st.table(df_loss_prob.style.set_properties(**{'text-align': 'center'}))

else:
    st.info("Use the parameters above and click **Run Simulation** to begin.")
