# A simple test of the Alpaca API
import alpaca_trade_api as tradeapi
import os

# source the environment variables
from dotenv import load_dotenv
load_dotenv()

# Set the API key and secret
# get these from your Alpaca account, set in .env file
api_key = 'your-key-here' # todo- why is dotenv not working :D
api_secret = 'your-secret-here'

# Create the Alpaca API object
api = tradeapi.REST(api_key, api_secret, api_version='v2')

# Get account information
account = api.get_account()
print(account.status)

