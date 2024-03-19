# app.py
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import pickle
import numpy as np

app = Flask(__name__)

# Load pickled data
with open('data_plot_bowling.pickle', 'rb') as f:
    df, _ = pickle.load(f)

# Data preparation
df.drop_duplicates(inplace=True)

@app.route('/')
def index():
    return render_template('indexbol.html')

@app.route('/result', methods=['POST'])
def result():
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

if __name__ == '__main__':
    app.run(debug=True)
