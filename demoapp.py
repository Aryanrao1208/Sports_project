from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
rf = pickle.load(open('football_score_predict.pkl', 'rb'))

@app.route('/index1.html')
def home():
    return render_template('index1.html')

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

if __name__ == '__main__':
    app.run(debug=True)
