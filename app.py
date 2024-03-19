# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np


from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


# app.py
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pickle
import numpy as np


# Load the Ridge Regression Classifier model
filename = 'ipl_score_predict_model.pkl'# change the model type to see lil changes in prediction
reg = pickle.load(open(filename, 'rb'))

# Load the trained model
rf = pickle.load(open('football_score_predict.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def redirect():
    return render_template('redirect.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/index1.html')
def index1():
    return render_template('index1.html')


@app.route('/indexbat.html')
def indexbat():
    return render_template('indexbat.html')

@app.route('/indexbol.html')
def indexbol():
    return render_template('indexbol.html')

@app.route('/indexcric.html')
def indexcric():
    return render_template('indexcric.html')

@app.route('/news/news1.html')
def news1():
    return render_template('/news/news1.html')

@app.route('/news/news2.html')
def news2():
    return render_template('/news/news2.html')

@app.route('/news/news3.html')
def news3():
    return render_template('/news/news3.html')

@app.route('/news/news4.html')
def news4():
    return render_template('/news/news4.html')

@app.route('/news/news5.html')
def news5():
    return render_template('/news/news5.html')

@app.route('/news/news6.html')
def news6():
    return render_template('/news/news6.html')

# @app.route('/news2.html')
# def news2():
#     return render_template('news2.html')

# @app.route('/news3.html')
# def news3():
#     return render_template('news3.html')

# @app.route('/news4.html')
# def news4():
#     return render_template('news4.html')



# ---------------------- this is the routing for the news seciton ----------------------------------

# @app.route('/news1.html')
# def news1():
#     return render_template('news1.html')

# @app.route('/news2.html')
# def news2():
#     return render_template('news2.html')

# @app.route('/news3.html')
# def news3():
#     return render_template('news3.html')

# @app.route('/news4.html')
# def news4():
#     return render_template('news4.html')

# @app.route('/news5.html')
# def news5():
#     return render_template('news5.html')

# @app.route('/news6.html')
# def news6():
#     return render_template('news6.html')

# @app.route('/news7.html')
# def news7():
#     return render_template('news7.html')

# @app.route('/news8.html')
# def news8():
#     return render_template('news8.html')


# @app.route('/')
# def home():
#     return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
        
        data = np.array([temp_array])
        my_prediction = int(reg.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)
    










@app.route('/predict1', methods=['POST'])
def predict1():
    # Get input values from the form
    team = request.form['team']
    opponent = request.form['opponent']
    venue = request.form['venue']
    time = request.form['time']
    date = request.form['date']
    
    # Preprocess the input data
    input_data = preprocess_input(team, opponent, venue, time, date)

    # Make predictions
    prediction = rf.predict(input_data)

    # Render the prediction result
    return render_template('result1.html', prediction=prediction)

def preprocess_input(team, opponent, venue, time, date):
    # Convert categorical variables to numerical or one-hot encode them
    team_code = get_team_code(team)
    opponent_code = get_team_code(opponent)
    venue_code = get_venue_code(venue)
    time = int(time.split(':')[0])  # Extract hours from the time string
    year, month, day = map(int, date.split('-'))  # Parse the date string into year, month, day components
    
    # Additional features
    # Calculate the goal difference between the team and opponent in previous matches
    prev_goal_diff = 0  # Replace with actual calculation
    
    # Calculate the win rate of the team in the last 5 matches
    win_rate = 0.5  # Replace with actual calculation
    
    # Encode the match outcome (W/D/L) of the previous match
    prev_match_outcome = 1  # Replace with actual encoding
    
    # Calculate the average goals scored by the team in the last 3 matches
    avg_goals_scored = 1.5  # Replace with actual calculation
    
    # Calculate the average goals conceded by the team in the last 3 matches
    avg_goals_conceded = 1.0  # Replace with actual calculation
    
    # Create an array with the additional features
    input_data = np.array([[team_code, opponent_code, venue_code, time, year, month, day,
                            prev_goal_diff, win_rate, prev_match_outcome,
                            avg_goals_scored, avg_goals_conceded]])
    
    return input_data


def get_team_code(team):
    # Implement logic to get the code for the team
    # For example, you can use a dictionary to map team names to codes
    # Here's a simple example:
    team_codes = {'Manchester City': 1, 'Chelsea': 2, 'Arsenal': 3, 'Tottenham Hotspur': 4, 'Manchester United': 5,
                  'West Ham United': 6, 'Wolverhampton Wanderers': 7, 'Newcastle United': 8}
    return team_codes.get(team, 0)  # Return 0 if team not found

def get_venue_code(venue):
    # Implement logic to get the code for the venue
    # For example, you can use a dictionary to map venue names to codes
    # Here's a simple example:
    venue_codes = {'Home': 1, 'Away': 2}
    return venue_codes.get(venue, 0)  # Return 0 if venue not found



# ------------------------------------------------------------------appy.pycode----------------------------------------------------------------











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

# # Importing necessary libraries
# import seaborn as sns
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64

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





# @app.route('/')
# def index():
#     return render_template('indexbat.html')

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





# -------------------------------------------------------------------appynew.pycode--------------------------------------------------------------



# Load pickled data
with open('data_plot_bowling.pickle', 'rb') as f:
    df, _ = pickle.load(f)

# Data preparation
df.drop_duplicates(inplace=True)

# @app.route('/')
# def index():
#     return render_template('indexbol.html')

@app.route('/resultbol', methods=['POST'])
def result_bol():
    players_of_interest = [request.form['player1'], request.form['player2'], request.form['player3'], request.form['player4'], request.form['player5']]
    
    # Plot 1: Total Matches Played
    temp1 = df.groupby('Player')['Mat'].sum().reset_index()
    temp1.columns = ['Player', 'Total Matches Played']
    temp1 = temp1[temp1['Player'].isin(players_of_interest)].sort_values('Total Matches Played', ascending=False)
    plot1 = plot_bar(temp1, 'Player', 'Total Matches Played', 'Total Matches Played by Selected Players')
    
    # Plot 2: Total Runs Conceded
    temp2 = df.groupby('Player')['Runs'].sum().reset_index()
    temp2.columns = ['Player', 'Total Runs Conceded']
    temp2 = temp2[temp2['Player'].isin(players_of_interest)].sort_values('Total Runs Conceded', ascending=False)
    plot2 = plot_bar(temp2, 'Player', 'Total Runs Conceded', 'Total Runs Conceded by Selected Players')
    
    # Plot 3: Total Wickets Taken
    temp3 = df.groupby('Player')['Wkts'].sum().reset_index()
    temp3.columns = ['Player', 'Total Wickets Taken']
    temp3 = temp3[temp3['Player'].isin(players_of_interest)].sort_values('Total Wickets Taken', ascending=False)
    plot3 = plot_strip(temp3, 'Total Wickets Taken', 'Player', 'Total Wickets Taken by Selected Players')
    
    # Plot 4: Total Economy Rate
    temp4 = df.groupby('Player')['Econ'].first().reset_index()
    temp4 = temp4[temp4['Player'].isin(players_of_interest)].sort_values('Econ', ascending=False)
    plot4 = plot_radar(temp4, 'Player', 'Econ', 'Total Economy Rate of Selected Players')
    
    # Plot 5: Strike Rate
    temp5 = df.groupby('Player')['SR'].first().reset_index()
    temp5.columns = ['Player', 'Strike Rate']
    temp5 = temp5[temp5['Player'].isin(players_of_interest)].sort_values('Strike Rate', ascending=False)
    plot5 = plot_point(temp5, 'Player', 'Strike Rate', 'Strike Rate of Selected Players')
    
    return render_template('resultbol.html', plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4, plot5=plot5)

def plot_bar(data, x_col, y_col, title):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x=x_col, y=y_col, palette='Paired')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def plot_strip(data, x_col, y_col, title):
    plt.figure(figsize=(8, 6))
    sns.stripplot(data=data, x=x_col, y=y_col, palette='tab10', size=10, jitter=True)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def plot_radar(data, x_col, y_col, title):
    plt.figure(figsize=(8, 6))
    angles = [i / float(len(data)) * 2 * np.pi for i in range(len(data))]
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, data[y_col], color='skyblue', alpha=0.25)
    ax.plot(angles, data[y_col], color='skyblue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(data[x_col], fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data

def plot_point(data, x_col, y_col, title):
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=data, x=x_col, y=y_col, color='blue', markers='o')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return plot_data


# --------------------------------------------------------cricketmatchresult----------------------------------------------------------------




# Load the trained model
with open('modelcricket_result.pkl', 'rb') as f:
    model = pickle.load(f)

# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/predictcric', methods=['POST'])
def predictcric():
    # Get user input from the form
    team1 = request.form['team1']
    team2 = request.form['team2']
    city = request.form['city']
    season = int(request.form['season'])
    toss_winner = request.form['toss_winner']
    toss_decision = request.form['toss_decision']
    
    # Map team and city names to numerical values (if needed)
    # You may need to adjust this mapping based on the encoding used in your model
    team_mapper = {"Mumbai Indians": 0, "Chennai Super Kings": 1, "Kolkata Knight Riders": 2}  # Add other teams as needed
    city_mapper = {"Mumbai": 0, "Chennai": 1, "Kolkata": 2}  # Add other cities as needed
    toss_decision_mapper = {"bat": 0, "field": 1}
    
    # Convert user inputs to numerical values
    team1_encoded = team_mapper.get(team1)
    team2_encoded = team_mapper.get(team2)
    city_encoded = city_mapper.get(city)
    toss_winner_encoded = team_mapper.get(toss_winner)
    toss_decision_encoded = toss_decision_mapper.get(toss_decision)
    
    # Make prediction
    prediction = model.predict([[team1_encoded, team2_encoded, city_encoded, season, toss_winner_encoded, toss_decision_encoded]])[0]
    
    # Map prediction back to team name
    teams = {0: "Mumbai Indians", 1: "Chennai Super Kings", 2: "Kolkata Knight Riders"}  # Add other teams as needed
    predicted_winner = teams.get(prediction, "Unknown")
    
    return render_template('resultcric.html', prediction=predicted_winner)









if __name__ == '__main__':
    # Change the port number to 7070
    app.run(debug=True)


