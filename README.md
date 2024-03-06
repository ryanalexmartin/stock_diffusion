This is a bot that uses a Latent Diffusion Model (LDM) to encode lots of different data and its time domain (news, unemployment rates, etc)
into a latent space, and attempts to use a Latent Diffusion Model to predict the behavior of stocks based on that information.

Basically, we are taking "Stable Diffusion" and applying it to the stock market.


# Input data
- Price and Volume Data
- Technical Indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- Fundamental Analysis Data (Revenue, profit margins, EPS, etc)
- Sentiment Analysis Data (Sentiment analysis of news headlines, social media posts, and analysts' reports)
    * Sentiment scores, often ranging from negative to positive, can be computed using NLP techniques and used as features.
- Market Indices and Economic Indicators (GDP growth rates, unemployment rates, interest rates)
    - Interest rates are a very important indicator.  If interest rates are high, the stock market as a whole is likely to go down.  If interest rates are low, the stock market as a whole is likely to go up.
    - Inflation rates, consumer confidence, manufacturing output, etc.
- Event and Anomaly Data (Significant corporate or economic events)
- Alternative Data (satellite imagery, credit card transaction volumes, web traffic data)
- POLITICIANS PUBLICLY AVAILABLE STOCK TRADES
    - This is a very important data source.  If a politician is buying a stock, it is likely to go up.  If a politician is selling a stock, it is likely to go down.
    - This data is publicly available, and can be used to make informed trading decisions.
    - A few examples of politicians who have made a lot of money trading stocks are Nancy Pelosi, and many others.

# Output data
We do not want to simply output the price of a stock.  This will not be useful.  Instead, we want to output a probability distribution
of the stock price at a given time.  This will allow us to make more informed decisions about the stock market.

# Model
- Latent Diffusion Model (LDM)

An introduction to the Latent Diffusion Model (LDM) and its application to stock trading:

The Latent Diffusion Model (LDM) is a powerful framework for learning representations of complex, high-dimensional data. In the context of 
stock trading, the LDM can be used to encode a wide range of information sources (e.g., price and volume data, technical indicators, 
fundamental analysis, sentiment analysis, market indices, economic indicators, event and anomaly data, alternative data) into a latent 
space that captures meaningful patterns and relationships. This latent space representation can then be leveraged to make predictions 
about stock prices and market behavior.

The core principle of diffusion models involves a forward diffusion process that gradually adds noise to the data over time, and a 
reverse process that learns to reconstruct the original data from the noisy version. In the context of stock trading, the forward
process can be defined as a stochastic process that models the evolution of the latent space over time, capturing the dynamics of
market conditions and signals. The reverse process aims to learn a mapping from the latent space back to the original data, enabling
the model to generate predictions and make inferences about future stock prices.

By training the LDM on historical stock data and a diverse set of information sources, the model can learn to extract relevant features
and patterns from the data, effectively capturing the complex interactions and dependencies that drive stock market behavior. This
enables the LDM to make accurate predictions and provide valuable insights for trading and investment strategies.

## The basic structure of the LDM:

1. Data Normalization:
A common normalization technique is the Min-Max Scaling, which adjusts the data to a common scale without distorting differences in the ranges of values.

2. Feature Engineering for Financial Data:
Example: Calculating Moving Average.
Where:
    (MA_t) is the moving average at time (t),
    (P_{t-i}) is the price at time (t-i),
    (k) is the number of periods to calculate the average over.

3. Latent Diffusion Process:
The forward process can be defined as:
[ q(\mathbf{x}t|\mathbf{x}{t-1}) = \mathcal{N}(\mathbf{x}t; \sqrt{1-\beta_t}\mathbf{x}{t-1}, \beta_t\mathbf{I}) ]

And the reverse process aims to learn:
[ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) ]

Where:
    (\mathbf{x}_t) is the data at step (t),
    (\beta_t) is the variance of the noise added at step (t),
    (\mathcal{N}) denotes the normal distribution,
    (\mathbf{I}) is the identity matrix.


## The data format used in the LDM:

The data used in the LDM is typically structured as a time series, with each data point representing a specific time period (e.g., daily, hourly, etc.). 
The data can include multiple features, such as price and volume data, technical indicators, fundamental analysis metrics, sentiment scores, economic 
indicators, and more. Each feature is associated with a specific time stamp, allowing the model to capture temporal dependencies and patterns in the data.

We need a consistent file format for the data, so for each type of indicator, we can use a CSV file with the following structure:

```
Date,Feature1,Feature2,Feature3,...
2022-01-01,Value1,Value2,Value3,...
2022-01-02,Value1,Value2,Value3,...
...
```

Where:
- `Date` is the timestamp associated with the data point
- `Feature1, Feature2, Feature3, ...` are the different features or indicators.  For example, for price and volume data, these could be Open, High, Low, Close, Volume. For technical indicators, these could be Moving Average, RSI, MACD, Bollinger Bands, etc.
- `Value1, Value2, Value3, ...` are the corresponding values for each feature at the given timestamp.

The data for each feature should be preprocessed and normalized as needed before being used as input to the LDM.

## Encoding the data in the Latent Space:

Since we are dealing with limitations of VRAM, we need to encode the data in the latent space.  This involves selecting and encoding a variety of data 
sources that capture relevant market signals and conditions. The choice of data significantly influences the model's ability to learn meaningful patterns 
and make accurate predictions. 

It would be impossible to include an entire news article, for example, in the latent space.  Instead, we can use sentiment analysis to extract a sentiment
score from the article, and use that as a feature in the latent space.  
Interestingly, the subject the article pertains to can also be used as a feature in the latent space.  For example, if the article is about a company,
the company name can be used as a feature in the latent space.  The characteristics, industry, and other information about that company, not its name,
will be the features in the latent space.

There are many companies, and many news articles, and many other data sources for any given day.  We need to encode all of this information into a
latent space that is small enough to fit into VRAM, but large enough to capture the relevant information.  This is a challenging problem, and the
choice of data and the encoding process is crucial to the success of the model.

We propose to use a combination of the following data sources to encode the latent space:

1. Price and Volume Data:
    - Open, High, Low, Close, Volume
    - Can be trivially obtained from financial data providers such as Yahoo Finance, Alpha Vantage, or other sources.
    - This data is crucial for capturing the historical price movements and trading volumes of stocks.
    - It will be encoded in the latent space as a time series of price and volume data. (filename: price_volume.csv)

2. Technical Indicators:
    - Moving Averages, RSI, MACD, Bollinger Bands
    - Can be calculated from price and volume data using standard formulas.
    - These indicators capture important signals about the momentum, volatility, and trend of stock prices.
    - They will be encoded in the latent space as a time series of technical indicator values. (filename: technical_indicators.csv)

3. Fundamental Analysis Data:
    - Revenue, profit margins, EPS, etc
    - Can be obtained from financial statements and reports of publicly traded companies.
    - These metrics provide insights into the financial health and performance of companies.
    - They will be encoded in the latent space as a time series of fundamental analysis metrics. (filename: fundamental_analysis.csv)

4. Sentiment Analysis Data:
    - Sentiment scores, often ranging from negative to positive, can be computed using NLP techniques and used as features.
    - Can be obtained from news headlines, social media posts, and analysts' reports.
    - These scores capture market sentiment and investor sentiment about specific stocks and market conditions.
    - They will be encoded in the latent space as a time series of sentiment scores. (filename: sentiment_analysis.csv)

5. Market Indices and Economic Indicators:
    - GDP growth rates, unemployment rates, interest rates
    - Can be obtained from government and economic data sources.
    - These indicators provide insights into broader economic conditions and trends.
    - They will be encoded in the latent space as a time series of economic indicators. (filename: economic_indicators.csv)

6. Event and Anomaly Data:
    - Significant corporate or economic events
    - Can be obtained from news sources and event databases.
    - These events can have significant impacts on stock prices and market behavior.
    - Example: Earnings reports, product launches, regulatory changes, natural disasters, etc. 
    - We will capture these events as binary indicators (0 or 1) for each day, indicating whether a significant event or anomaly occurred.
    - This will be determined based on a simple criterion:  Was the news volume for a given day significantly higher than the average news volume?
    - They will be encoded in the latent space as a time series of event and anomaly data. (filename: event_anomaly.csv)

7. Alternative Data:
    - Satellite imagery, credit card transaction volumes, web traffic data
    - Can be obtained from alternative data providers and sources.
    - These data sources provide unique and non-traditional signals about market conditions and trends.
    - Satellite imagery can provide insights into physical infrastructure, economic activity, and environmental conditions.
    - Credit card transaction volumes and web traffic data can provide insights into consumer behavior and economic activity.
    - They will be encoded in the latent space as a time series of alternative data. (filename: alternative_data.csv)

The data from each of these sources will be preprocessed and normalized as needed before being used as input to the LDM. The preprocessed data will then be
encoded into the latent space using the LDM, capturing the relevant signals and patterns that drive stock market behavior.



## How the latent space encodings work

The LDM is a generative model that learns a latent representation of the input data and captures the underlying structure and patterns in the data.
The model consists of two main components: the encoder and the decoder.

The encoder takes the input data and maps it to a latent space, where the data is represented as a set of latent variables that capture the relevant 
features and patterns in the data. The encoder learns to extract meaningful representations of the input data, effectively capturing the complex 
interactions and dependencies that drive stock market behavior.

The decoder takes the latent representation and maps it back to the original data space, effectively reconstructing the input data from the latent
variables. The decoder learns to generate predictions and make inferences about future stock prices based on the latent representation, leveraging
the learned patterns and features to make accurate predictions.

The latent space encodings can be visualized as a low-dimensional representation of the input data, capturing the relevant signals and patterns that
drive stock market behavior. The latent space encodings effectively summarize the input data in a compact and informative manner, enabling the model
to make accurate predictions and provide valuable insights for trading and investment strategies.


### A mathematical breakdown:

The encoder maps the input data (\mathbf{x}) to a latent space representation (\mathbf{z}):

[ q(\mathbf{z}|\mathbf{x}) ]

The decoder maps the latent representation (\mathbf{z}) back to the original data space (\mathbf{x}):

[ p_\theta(\mathbf{x}|\mathbf{z}) ]

Where:
- (\mathbf{x}) is the input data
- (\mathbf{z}) is the latent space representation
- (q) is the encoder distribution
- (p_\theta) is the decoder distribution
- (\theta) are the parameters of the decoder

The encoder and decoder are trained jointly to learn the optimal latent representation of the input data, effectively capturing the relevant features
and patterns that drive stock market behavior. The learned latent representation can then be used to make predictions and provide valuable insights
for trading and investment strategies.


## Training the LDM:

Using PyTorch:

`python3 -m pip install torch torchvision torchaudio`

`python3 train.py --data_dir /path/to/data --output_dir /path/to/output --num_epochs 100 --batch_size 64 --learning_rate 0.001`

Where:
- `--data_dir` is the path to the directory containing the preprocessed and normalized data
- `--output_dir` is the path to the directory where the trained model and results will be saved
- `--num_epochs` is the number of training epochs
- `--batch_size` is the batch size for training
- `--learning_rate` is the learning rate for training


## Example dataset and expected format:

Filesystem tree:
```
data/
    company_stock_1/
        price_volume.csv
        technical_indicators.csv
        fundamental_analysis.csv
        sentiment_analysis.csv
        economic_indicators.csv
        event_anomaly.csv
        alternative_data.csv
    company_stock_2/
        price_volume.csv
        technical_indicators.csv
        fundamental_analysis.csv
        sentiment_analysis.csv
        economic_indicators.csv
        event_anomaly.csv
        alternative_data.csv
    ...
```

As you can see, each company's data is stored in a separate directory, and each data source is stored in a separate CSV file within that directory.
This allows us to easily organize and manage the data for different companies and apply the LDM to each company's stock data separately.

The data for each company should be preprocessed and normalized as needed before being used as input to the LDM. The preprocessed data will then be
encoded into the latent space using the LDM, capturing the relevant signals and patterns that drive stock market behavior.


## Generating predictions from the LDM:

Using the trained model:

`python3 predict.py --model_path /path/to/model --input_data /path/to/input_data --output_dir /path/to/output`

Where:
- `--model_path` is the path to the trained LDM model
- `--input_data` is the path to the input data for which predictions are to be generated
- `--output_dir` (optional) is the path to the directory where the predictions will be saved.

Example console output:
```
Predicting stock prices using Latent Diffusion Model...
Generating predictions for input data...

Using information obtained from today's date 2024-01-01
company_stock_1:
    Predicted stock price distribution for 2024-01-02:
        Mean: $100.00
        Standard Deviation: $5.00
        Confidence Interval: [95.00, 105.00]
    Predicted stock price distribution for 2024-01-03:
        Mean: $102.00
        Standard Deviation: $5.00
        Confidence Interval: [97.00, 107.00]

company_stock_2:
    Predicted stock price distribution for 2024-01-02:
        Mean: $150.00
        Standard Deviation: $7.00
        Confidence Interval: [143.00, 157.00]
    Predicted stock price distribution for 2024-01-03:
        Mean: $155.00
        Standard Deviation: $7.00
        Confidence Interval: [148.00, 162.00]

Predicting an increase of 2% in stock price for company_stock_1 on 2024-01-03
Predicting an increase of 3% in stock price for company_stock_2 on 2024-01-03
```


# Autotrade

The ultimate goal of this project is to create a bot that can automatically trade stocks based on the predictions generated by the LDM.
The bot will use the predicted stock price distributions to make informed trading decisions, leveraging the insights and predictions provided
by the LDM to execute buy and sell orders for stocks.

The autotrade bot will be designed to:
- Monitor the predicted stock price distributions for different stocks
- Scrape web sources for news, social media, and other relevant information
- Identify trading opportunities based on the predicted price movements
- Execute buy and sell orders for stocks based on the predictions and market conditions
- Manage risk and portfolio allocation based on the predicted confidence intervals and market volatility
- Adapt to changing market conditions and update trading strategies based on new information and predictions


### How to run the autotrade bot:

`python3 autotrade.py --model_path /path/to/model -c /path/to/config.yml`

In addition to the output of the LDM, the autotrade bot will use other mathematical formulas to determine the best course of action for the stocks.

- Dynamic Hedging: The bot will dynamically hedge the portfolio based on the predicted confidence intervals and market volatility.  For example, if the
predicted confidence interval is wide and the market volatility is high, the bot may reduce the allocation to the stock or implement a stop-loss strategy
to manage risk.  If the predicted confidence interval is narrow and the market volatility is low, the bot may increase the allocation to the stock or
implement a momentum trading strategy to capitalize on potential price movements.
- Mean Reversion: The bot will use mean reversion trading strategies to capitalize on short-term price movements and market inefficiencies.  For example,
if the predicted stock price distribution indicates a high probability of a price increase, the bot may execute a buy order to take advantage of the
expected price movement.  If the predicted stock price distribution indicates a high probability of a price decrease, the bot may execute a sell order
to capitalize on the expected price movement.
- Momentum Trading: The bot will use momentum trading strategies to capitalize on longer-term price trends and market momentum.  For example, if the
predicted stock price distribution indicates a high probability of a price increase over a longer time horizon, the bot may execute a buy order to take
advantage of the expected price movement.  If the predicted stock price distribution indicates a high probability of a price decrease over a longer time
horizon, the bot may execute a sell order to capitalize on the expected price movement.
- Stop-Loss: The bot will use stop-loss strategies to manage risk and protect the portfolio from potential losses.  For example, if the predicted stock
price distribution indicates a high probability of a price decrease, the bot may implement a stop-loss order to limit potential losses and protect the
portfolio from adverse price movements.
- Black-Scholes Model: The bot will use the Black-Scholes model to calculate the theoretical price of options and derivatives, and use this information
to make informed trading decisions.  For example, the bot may use the Black-Scholes model to calculate the implied volatility of options and use this
information to hedge the portfolio or execute trading strategies based on the predicted price movements.

In the config file, there will be a list of stocks to monitor, and many other options such as the amount of money to allocate to each stock, the
news sources to scrape, the trading strategy to use, and more.  Here's an example:

```
stocks:
    - symbol: AAPL
      allocation: 10000
      news_sources:
          - Yahoo Finance
          - Bloomberg
          - CNBC
      trading_strategy: mean_reversion
      risk_management: volatility_adjusted
    - symbol: MSFT
      allocation: 5000
      news_sources:
          - Yahoo Finance
          - Bloomberg
          - CNBC
      trading_strategy: momentum
      risk_management: stop_loss
    ...
```

Or, for the cryptocurrency market:

```
cryptocurrencies:
    - symbol: BTC
      allocation: 10000
      news_sources:
          - CoinDesk
          - CryptoSlate
          - Cointelegraph
      trading_strategy: mean_reversion
      risk_management: volatility_adjusted
    - symbol: ETH
      allocation: 5000
      news_sources:
          - CoinDesk
          - CryptoSlate
          - Cointelegraph
      trading_strategy: momentum
      risk_management: stop_loss
    ...
```


### Black-Scholes Model
[ C(S,t) = N(d_1)S - N(d_2)K e^{-r(T-t)} ]

Where:
- (C) is the theoretical price of the option
- (S) is the current stock price
- (t) is the time to expiration
- (N) is the cumulative distribution function of the standard normal distribution
- (d_1) and (d_2) are the parameters of the Black-Scholes model
- (K) is the strike price of the option
- (r) is the risk-free interest rate
- (T) is the time to expiration


### Making the Black-Scholes Model work with the LDM

The output of our LDM is a probability distribution of the stock price at a given time.  We can use this information to calculate the theoretical
price of options and derivatives using the Black-Scholes model.  For example, if the predicted stock price distribution indicates a high probability
of a price increase, the bot may use the Black-Scholes model to calculate the theoretical price of call options and use this information to make
informed trading decisions.  If the predicted stock price distribution indicates a high probability of a price decrease, the bot may use the
Black-Scholes model to calculate the theoretical price of put options and use this information to make informed trading decisions.

Mathematically, the LDM output can be represented as:

[ p(\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_t; \mu_t, \Sigma_t) ]

Where:
- (\mathbf{x}_t) is the stock price at time (t)
- (\mu_t) is the mean of the predicted stock price distribution
- (\Sigma_t) is the covariance matrix of the predicted stock price distribution
- (\mathcal{N}) denotes the multivariate normal distribution



The best way to combine the LDM with the Black-Scholes model is to use the predicted stock price distribution as input to the Black-Scholes model:

That is, we can use the mean (\mu_t) and standard deviation (\Sigma_t) of the predicted stock price distribution as input to the Black-Scholes model,
and use this information to calculate the theoretical price of options and derivatives.  This allows us to leverage the insights and predictions
provided by the LDM to make informed trading decisions and manage risk in the options and derivatives market.

Mathematically, we can represent the Black-Scholes model with the predicted stock price distribution as:

[ C(\mu_t, \Sigma_t) = N(d_1)S - N(d_2)K e^{-r(T-t)} ]

Where:
- (C(\mu_t, \Sigma_t)) is the theoretical price of the option based on the predicted stock price distribution from the LDM
- (S) is the current stock price
- (N) is the cumulative distribution function of the standard normal distribution
- (d_1) and (d_2) are the parameters of the Black-Scholes model
- (K) is the strike price of the option
- (r) is the risk-free interest rate
- (T) is the time to expiration

By using the predicted stock price distribution as input to the Black-Scholes model, we can make informed trading decisions and manage risk in the
options and derivatives market based on the insights and predictions provided by the LDM.




