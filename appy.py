from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = Flask(__name__)

def import_data(file_path):
    df = pd.read_csv(file_path, index_col=[0])
    return df

def data_preparation(df):
    df.drop_duplicates(inplace=True)
    return df

def plot_seasons_played(df, players_of_interest):
    sns.set_context('notebook', font_scale=1.25)
    no_of_seasons = df['Player'].value_counts().reset_index()
    no_of_seasons.columns = ['Player', 'No. of Seasons Played']
    no_of_seasons.sort_values('Player', inplace=True)
    no_of_seasons.reset_index(inplace=True)
    
    # Filter dataframe based on all players of interest
    filtered_temp = no_of_seasons[no_of_seasons['Player'].isin(players_of_interest)]
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=filtered_temp, x='Player', y='No. of Seasons Played', palette='tab10')
    plt.title('Number of Seasons Played by Selected Players')
    plt.xlabel('Player')
    plt.ylabel('Number of Seasons Played')
    plt.xticks(rotation=15)
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image data as base64 string
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()  # Close the plot
    
    return plot_data

def plot_total_matches_played(df, players_of_interest):
    sns.set_context('notebook', font_scale=1.25)
    matches = df.groupby('Player')['Mat'].sum().reset_index()
    matches.columns = ['Player', 'Total Matches Played']
    temp = matches[matches['Player'].isin(players_of_interest)].sort_values('Total Matches Played', ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=temp, x='Total Matches Played', y='Player', palette='Paired')
    plt.title('Total Matches Played by Selected Players')
    plt.xlabel('Total Matches Played')
    plt.ylabel('Player')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plot_data1 = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()
    
    return plot_data1


def plot_total_runs_scored(df, players_of_interest):
    sns.set_context('notebook', font_scale=1.25)
    runs_s = df.groupby('Player')['Runs'].sum().reset_index()
    runs_s.columns = ['Player', 'Total Runs']
    runs_filtered = runs_s[runs_s['Player'].isin(players_of_interest)]

    plt.figure(figsize=(6, 4))
    sns.pointplot(data=runs_filtered, x='Player', y='Total Runs', color='b', markers='o')
    plt.title('Total Runs Scored by Selected Players')
    plt.xlabel('Player')
    plt.ylabel('Total Runs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plot_data2 = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()
    
    return plot_data2


    

def plot_fifties_scored(df, players_of_interest):
    # Assuming fifties_s is a DataFrame containing statistics of 50s scored
    fifties_s = df.groupby('Player')['50'].sum().reset_index()
    fifties_s.columns = ['Player', "Number of 50's"]

    # Filter the data for the top 50 players with the most 50s
    temp = fifties_s.sort_values("Number of 50's", ascending=False)[:50]

    # Filter the data for the players of interest
    temp_players_of_interest = temp[temp['Player'].isin(players_of_interest)]

    # Sorting and plotting the data for players of interest
    plt.figure(figsize=(6, 4))
    sns.barplot(data=temp_players_of_interest, x='Player', y="Number of 50's", palette='Set1')
    plt.title("Number of 50's Scored by Selected Players")
    plt.xlabel('Player')
    plt.ylabel("Number of 50's")
    plt.xticks(rotation=80)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image data as base64 string
    plot_data = base64.b64encode(buf.read()).decode('utf-8')

    plt.close()  # Close the plot

    return plot_data

def plot_hundreds_scored(df, players_of_interest):
    # Assuming hundreds_s is a DataFrame containing statistics of 100s scored
    hundreds_s = df.groupby('Player')['100'].sum().reset_index()
    hundreds_s.columns = ['Player', "Number of 100's"]

    # Filter the data for the top 50 players with the most 100s
    temp = hundreds_s.sort_values("Number of 100's", ascending=False)[:50]

    # Filter the data for the players of interest
    temp_players_of_interest = temp[temp['Player'].isin(players_of_interest)]

    # Sorting and plotting the data for players of interest
    plt.figure(figsize=(6, 4))
    sns.barplot(data=temp_players_of_interest, x='Player', y="Number of 100's", palette='Set1')
    plt.title("Number of 100's Scored by Selected Players")
    plt.xlabel('Player')
    plt.ylabel("Number of 100's")
    plt.xticks(rotation=80)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image data as base64 string
    plot_data = base64.b64encode(buf.read()).decode('utf-8')

    plt.close()  # Close the plot

    return plot_data

# Importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_fours_scored(df, players_of_interest):
    # Filter the data for the top 50 players with the most 4s
    fours_s = df.groupby('Player')['4s'].sum().reset_index()
    fours_s.columns = ['Player', "Number of 4's"]
    temp = fours_s.sort_values("Number of 4's", ascending=False)[:50]

    # Filter the data for the players of interest
    temp_players_of_interest = temp[temp['Player'].isin(players_of_interest)]

    # Create a dot chart using stripplot
    plt.figure(figsize=(6,4))
    sns.stripplot(data=temp_players_of_interest, x='Player', y="Number of 4's", palette='Set1', jitter=True, size=10)
    plt.title("Number of 4's Scored by Selected Players")
    plt.xlabel('Player')
    plt.ylabel("Number of 4's")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image data as base64 string
    plot_data = base64.b64encode(buf.read()).decode('utf-8')

    plt.close()  # Close the plot

    return plot_data

def plot_sixes_scored(df, players_of_interest):
    # Filter the data for the top 50 players with the most 6s
    sixes_s = df.groupby('Player')['6s'].sum().reset_index()
    sixes_s.columns = ['Player', "Number of 6's"]
    temp = sixes_s.sort_values("Number of 6's", ascending=False)[:50]

    # Filter the data for the players of interest
    temp_players_of_interest = temp[temp['Player'].isin(players_of_interest)]

    # Create a dot chart using stripplot
    plt.figure(figsize=(6,4))
    sns.stripplot(data=temp_players_of_interest, x='Player', y="Number of 6's", palette='Set1', jitter=True, size=10)
    plt.title("Number of 6's Scored by Selected Players")
    plt.xlabel('Player')
    plt.ylabel("Number of 6's")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image data as base64 string
    plot_data6 = base64.b64encode(buf.read()).decode('utf-8')

    plt.close()  # Close the plot

    return plot_data6













@app.route('/')
def index():
    return render_template('indexbat.html')

@app.route('/result', methods=['POST'])
def result():
    players_of_interest = [request.form['player1'], request.form['player2'],
                           request.form['player3'], request.form['player4'],
                           request.form['player5']]
    file_path = 'C:/Users/HOME/OneDrive/Desktop/removed project/ipl stats/Most Runs All Seasons Combine.csv'
    df = import_data(file_path)
    df = data_preparation(df)
    
    plot_data = plot_seasons_played(df, players_of_interest)
    plot_data1 = plot_total_matches_played(df, players_of_interest)
    plot_data2 = plot_total_runs_scored(df, players_of_interest)
    plot_data3 = plot_fifties_scored(df, players_of_interest)
    plot_data4 = plot_hundreds_scored(df, players_of_interest)  # Call plot_hundreds_scored
    plot_data5 = plot_fours_scored(df, players_of_interest)  # Call plot_fours_scored
    plot_data6 = plot_sixes_scored(df, players_of_interest)  # Call plot_sixes_scored
    
    return render_template('resultbat.html', plot_data_seasons_played=plot_data,
                           plot_data_total_matches_played=plot_data1,
                           plot_data_total_runs_scored=plot_data2,
                           plot_data_fifties_scored=plot_data3,
                           plot_data_hundreds_scored=plot_data4,
                           plot_data_fours_scored=plot_data5,
                           plot_data_sixes_scored=plot_data6)  # Pass plot_data6 to template

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)