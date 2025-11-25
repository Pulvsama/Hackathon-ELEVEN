import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('Understanding the data/waiting_times_train.csv', parse_dates=['DATETIME'])
train_df["DATETIME"] = pd.to_datetime(train_df["DATETIME"])
train_df_sorted = train_df.sort_values(by='DATETIME')

def plot_wait_times_train(attraction, year, month, day):
    attraction_df = train_df_sorted[train_df_sorted['ENTITY_DESCRIPTION_SHORT'] == attraction]
    filtered_df = attraction_df[(attraction_df['DATETIME'].dt.year == year) & (attraction_df['DATETIME'].dt.month == month) & (attraction_df['DATETIME'].dt.day == day)]
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['DATETIME'], filtered_df['WAIT_TIME_IN_2H'], label='Wait Time', color='blue', alpha=0.5)
    plt.title('Wait Times Over Time')
    plt.xlabel('DATETIME')
    plt.ylabel('Wait Time (minutes)')
    plt.legend()
    plt.grid(True)
    plt.show()