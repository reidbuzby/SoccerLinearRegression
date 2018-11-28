import csv
import sklearn.linear_model as skl
import random
import numpy as np


'''
Takes in the path to the team data csv file and returns the features as an array
'''
def readTeam_Data(path):

    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')

        featuresDescriptions = []
        x = []
        first = True
        for row in reader:
            if first:
                featuresDescriptions = row
                first = False
            else:
                x.append(row)

    return (x, featuresDescriptions)

'''
Takes in the path to the match data csv file and the previously gathered team data as an array

For each match, the home and away team information is found from the team_data array and appended as features

The final feature matrix X is returned
'''
def readMatch_Data(path, team_data):

    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')

        featuresDescriptions = []
        first = True
        x = []
        count = 1
        for row in reader:
            if first:
                featuresDescriptions = row
                first = False
            else:
                newRow = []
                for element in row[9:41]:
                    if element == '':
                        newRow.append(0.0)
                    else:
                        newRow.append(float(element))
                date = row[5]
                home_team_id = row[7]
                away_team_id = row[8]
                home = findTeamGivenID_and_Date(home_team_id, date, team_data)
                away = findTeamGivenID_and_Date(away_team_id, date, team_data)

                if not home or not away:
                    continue

                for i in range(4, len(home)):
                    newRow.append(home[i])

                for i in range(4, len(away)):
                    newRow.append(away[i])

                x.append(newRow)
                if len(newRow) == 49:
                    print(count)

            count += 1

    return (x, featuresDescriptions)

'''
Given a team's ID and a match date, data on that team at that time is returned.

However, because this is FIFA data the teams are only updated approx once a year,
so we gather the team data that is closest to the match date
'''
def findTeamGivenID_and_Date(id, date, team_data):

    teams = []
    for row in team_data:
        if row[2] == id:
            teams.append(row)

    if not teams:
        return None

    try:
        matchYear = int(str(date)[5:7])
    except:
        matchYear = int(str(date)[6:8])


    min = 1000000
    minTeam = []
    for team in teams:
        try:
            upDate = int(str(team[3])[2:4])
        except:
            upDate = int(str(team[3])[3:5])
        diff = matchYear - upDate
        if diff > 0:
            if diff < min:
                min = diff
                minTeam = team

    if not minTeam:
        max = -10000000
        maxTeam = []
        for team in teams:
            upDate = int(str(team[3])[2:4])
            diff = matchYear - upDate
            if diff > max:
                max = diff
                maxTeam = team
        return stringFeaturesToInts(maxTeam)


    return stringFeaturesToInts(minTeam)

'''
Changes the team features from strings to integers
'''
def stringFeaturesToInts(team):

    low = ['Slow', 'Little', 'Short', 'Safe', 'Deep', 'Press', 'Narrow']
    middle = ['Balanced', 'Normal', 'Mixed', 'Medium', 'Double', 'Normal']
    high = ['Fast', 'Lots', 'Long', 'Risky', 'High', 'Contain', 'Wide']

    off = ['Free Form', 'Cover']
    on = ['Organised', 'Offside Trap']

    newRow = []
    for feature in team[4:]:
        if feature in low:
            newRow.append(1.0)
        elif feature in middle:
            newRow.append(2.0)
        elif feature in high:
            newRow.append(3.0)
        elif feature in off:
            newRow.append(-1.0)
        elif feature in on:
            newRow.append(1.0)
        elif feature == '':
            newRow.append(0.0)
        else:
            newRow.append(float(feature))

    return newRow



'''
Normalizes features
'''
def featureNormalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    x_norm = (X - mu) / sigma
    return x_norm

'''
Generates a Y vector from an input X matrix
'''
def y_from_x(x):
    y = []
    for game in x:
        homeGoals = game[9]
        y.append(homeGoals)

    return y

'''
Runs stochastic gradient descent using SKLearn
'''
def learn(X):
    shuffledX = random.sample(X, len(X))

    training = featureNormalize(shuffledX[:16000])
    y = y_from_x(training)

    validation = shuffledX[16001:]
    validation_y = y_from_x(validation)

    model = skl.SGDRegressor(max_iter=5000, tol=1e-3, alpha=0.0001)

    learned = model.fit(training, y)

    score = learned.score(validation, validation_y)

    print(score)

    # predicted = learned.predict(validation)
    # for val in predicted:
    #     print(val)


teamData = readTeam_Data('Team_Attributes.csv')[0]
X = readMatch_Data('Match_Data.csv', teamData)[0]

learn(X)
