from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('modelcricket_result.pkl', 'rb') as f:
    model = pickle.load(f)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    
    return render_template('result.html', prediction=predicted_winner)

if __name__ == '__main__':
    app.run(debug=True)
