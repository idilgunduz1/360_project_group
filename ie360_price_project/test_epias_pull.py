from src.data_loader import load_price_data_from_epias

df = load_price_data_from_epias(days_back=30)
print(df.head())
print(df.tail())
print(df.shape)