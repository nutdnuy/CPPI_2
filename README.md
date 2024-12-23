# CPPI Monte Carlo Simulation App

This project is a **Constant Proportion Portfolio Insurance (CPPI)** Monte Carlo simulation tool. It uses a **Streamlit** web interface to allow users to interactively set parameters for a CPPI strategy and run multiple random simulations to observe how the portfolio value evolves over time. The application aims to help users understand how CPPI works as a dynamic strategy for downside risk protection while seeking upside potential.

## What is CPPI?

Constant Proportion Portfolio Insurance (CPPI) is an investment strategy that dynamically adjusts the allocation between a risky asset (e.g., equities) and a risk-free asset (e.g., treasury bills) based on a predefined "cushion." The cushion is determined by the difference between the current portfolio value and a designated floor value. By multiplying this cushion by a chosen "multiplier," the strategy decides how much to invest in the risky asset. As the portfolio value changes due to market movements, the allocation is periodically rebalanced according to these rules.

**Key concepts of CPPI:**
- **Floor:** A minimum acceptable portfolio value you do not want to breach.
- **Cushion:** The difference between the current portfolio value and the floor.
- **Multiplier (m):** The sensitivity factor that determines how aggressively the portfolio invests in the risky asset. A higher multiplier invests more in the risky asset when above the floor, but can also lead to greater risk.
- **Rebalancing:** The frequency with which the portfolio is adjusted back to the CPPI-prescribed allocation.

## Features

- **Parameter Controls:**  
  Set various parameters such as:
  - **Initial Start Wealth:** The starting amount of the portfolio.
  - **Number of Years:** The total investment horizon.
  - **Expected Return (mu):** The annualized expected return of the risky asset.
  - **Volatility (sigma):** The annualized volatility of the risky asset.
  - **Risk-Free Rate:** The annualized risk-free return.
  - **Rebalancings per Year:** How many times per year the portfolio is rebalanced according to the CPPI rules.
  - **Number of Scenarios:** How many Monte Carlo runs to simulate.
  - **Floor (fraction of initial):** The floor value expressed as a fraction of the initial wealth.
  - **Multiplier (m):** The CPPI multiplier.
  - **Zoom Y Axis:** Adjust scaling of the final graph for better visibility.

- **Real-Time Visualization:**  
  After running simulations, the tool displays:
  - Distribution of final portfolio values after the chosen time horizon.
  - Selected percentiles of final outcomes.
  - Individual scenario paths (optional) to visualize the trajectory of the portfolio over time.
  
- **Explainer Sections:**  
  Additional collapsible sections ("What is CPPI?", "About This Simulation", "Limitations") to help users understand the strategy, the model assumptions, and caveats.

## Requirements

- **Python 3.9+** (Recommended)
- **Packages:**
  - `streamlit` for the web interface
  - `numpy`, `pandas` for data handling
  - `matplotlib` or `plotly` for visualization (depending on the chosen plotting library)
  - `scipy` or other necessary scientific libraries for random draws and calculations
  
You can install the dependencies via a `requirements.txt` file similar to:

```txt
streamlit
numpy
pandas
matplotlib
scipy
```

Adjust or add other packages if your code requires them.

## How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/nutdnuy/CPPI_2.git]
   cd cppi-monte-carlo-simulation
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

   Replace `app.py` with the actual Python file that launches the Streamlit interface.

4. **Open in Browser:**
   Streamlit will provide a local URL (usually `http://localhost:8501`). Open it in your browser to access the app.

## How to Deploy

You can deploy this application to various hosting services that support Streamlit apps, such as **Streamlit Cloud**, **Heroku**, or **Azure Web Apps**. The simplest route is Streamlit Community Cloud:

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with your GitHub account.
3. Select the repository and branch containing your code.
4. Deploy the application with the provided interface.

Your app should become accessible via a shared public URL.

## About the Simulation

The underlying logic is a Monte Carlo simulation of portfolio returns. Each scenario draws random returns from a specified distribution (typically normal, using the given expected return and volatility). The CPPI rule is applied at each rebalancing interval, adjusting the allocation between the risky and risk-free assets based on the current cushion.

**Assumptions:**
- Returns are independently and identically distributed (i.i.d.) normal random variables (unless otherwise specified).
- Perfect liquidity and no transaction costs.
- Continuous or discrete rebalancing as per user settings.

These assumptions can be relaxed or modified in the underlying code if you want a more realistic setup.

## Limitations

- **Model Risk:** The simulation assumes normal returns and may not accurately reflect fat-tailed distributions or real market shocks.
- **No Transaction Costs:** CPPI strategies in practice incur costs for rebalancing, not accounted for here.
- **No Taxes or Slippage:** The model ignores taxation, slippage, or other real-world market frictions.
- **Static Parameters:** The parameters (mu, sigma, risk-free rate) remain constant over the entire simulation horizon.

## Contributing

Contributions are welcome! If you have ideas on how to improve the simulation, feel free to open an issue or submit a pull request. Some potential contributions include:

- Adding alternative return distributions (e.g., lognormal, historical bootstrapping).
- Implementing transaction costs.
- Incorporating leverage constraints or other portfolio constraints.
- Improving the visualization or adding more robust sensitivity analysis tools.

## License

This project is provided under the [MIT License](LICENSE.md). You are free to use, modify, and distribute the code as permitted by the license.

---

**Enjoy exploring the CPPI strategy with this Monte Carlo simulation tool!**



## Ref. 

- Albert Dorador ; Constrained Max Drawdown: a Fast and Robust Portfolio Optimization Approach
- OLGA BIEDOVA and VICTORIA STEBLOVSKAYA;  MULTIPLIER OPTIMIZATION FOR CONSTANT PROPORTION PORTFOLIO INSURANCE (CPPI) STRATEGY
- Paulo José Martins Jorge da Silva; PORTFOLIO INSURANCE STRATEGIES FRIEND ORFOE?
- Daniel Mantilla-García Research Associate, EDHEC-Risk Institute  Head of Research & Development, Koris International ; Growth Optimal Portfolio Insurance for Long-Term Investors
- Introduction to Portfolio Construction and Analysis with Python [https://www.coursera.org/learn/introduction-portfolio-construction-python?utm_source=gg&utm_medium=sem&utm_campaign=b2c_apac_coursera-plus_coursera_ftcof_subscription_arte_apr-24_dr_geo-set-3-multi_sem_rsa_gads_lg-all&utm_content=b2c&campaignid=21165289867&adgroupid=163685947107&device=c&keyword=coursera&matchtype=e&network=g&devicemodel=&adpostion=&creativeid=696974723648&hide_mobile_promo=&gad_source=1&gclid=CjwKCAiAmfq6BhAsEiwAX1jsZ7y5czJmondJ1ybgRVOHfMBjPHu1OZ0PnCTRwRZraKgxArnB7qUW6RoCwvUQAvD_BwE]







