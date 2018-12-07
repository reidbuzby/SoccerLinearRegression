import csv
import sklearn.linear_model as skl
import sklearn.preprocessing as skp
import sklearn.svm as svm
import random
import numpy as np
import sys
import io
import math
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

                # Create new feature for defensive score
                if home[18] is not None and home[20] is not None:
                    homeDefensiveScore = home[18] * home[20]
                    home.append(homeDefensiveScore)
                if away[18] is not None and away[20] is not None:
                    awayDefensiveScore = away[18] * away[20]
                    away.append(awayDefensiveScore)

                # These features decreased our output score:

                # # Create new feature for offensive score
                # if home[11] is not None and home[13] is not None and home[15] is not None:
                #     homeOffesiveScore = home[11] * home[13] * home[15]
                #     home.append(homeOffesiveScore)
                # if away[11] is not None and away[13] is not None and away[15] is not None:
                #     awayOffesiveScore = away[11] * away[13] * away[15]
                #     away.append(awayOffesiveScore)
                #
                # if home[4] is not None and home[8] is not None:
                #     homeBuildUpScore = home[4] * home[8]
                #     home.append(homeBuildUpScore)
                # if away[4] is not None and away[8] is not None:
                #     awayBuildUpScore = away[4] * away[8]
                #     away.append(awayBuildUpScore)

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
            newRow.append(1.0)
        elif feature in on:
            newRow.append(2.0)
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
Creates a Y vector by classifying a match as a home team win (1.0) home team loss (2.0) or tie (3.0)
'''
def classifyMatch(home, away):
    Y = []

    for i in range(len(home)):
        if home[i] > away[i]:
            Y.append(1.0)
        elif away[i] > home[i]:
            Y.append(2.0)
        else:
            Y.append(3.0)

    return Y


'''
Trains and runs our linear regression model
'''
def linearRegression(data):
    print('--------LINEAR REGRESSION OUTPUT--------\n')
    training = data[0]
    yHomeTraining = data[1]
    yAwayTraining = data[2]

    validation = data[3]
    yHomeValidation = data[4]
    yAwayValidation = data[5]

    test = data[6]
    yHomeTest = data[7]
    yAwayTest = data[8]

    homeModel = skl.SGDRegressor(max_iter=5000, tol=1e-3, alpha=0.001)
    homeLearned = homeModel.fit(training, yHomeTraining)

    awayModel = skl.SGDRegressor(max_iter=5000, tol=1e-3, alpha=0.001)
    awayLearned = awayModel.fit(training, yAwayTraining)

    homePredictTrain = homeModel.predict(training)
    awayPredictTrain = awayModel.predict(training)

    # homeScore = homeLearned.score(validation, yHomeValidation)
    # print('Home goals linear Regression score: ' + str(homeScore))
    # awayScore = awayLearned.score(validation, yAwayValidation)
    # print('Away goals linear Regression score: ' + str(awayScore))

    homeScore = homeLearned.score(test, yHomeTest)
    print('Home goals linear Regression score: ' + str(homeScore))
    awayScore = awayLearned.score(test, yAwayTest)
    print('Away goals linear Regression score: ' + str(awayScore))

    # homePredicted = homeLearned.predict(validation)
    # awayPredicted = awayLearned.predict(validation)

    homePredicted = homeLearned.predict(test)
    awayPredicted = awayLearned.predict(test)


    correct = 0
    awayWins = 0
    for i in range(len(homePredicted)):
        if yHomeTest[i] > yAwayTest[i] and homePredicted[i] > awayPredicted[i]:
            correct += 1
        elif yHomeTest[i] < yAwayTest[i] and homePredicted[i] < awayPredicted[i]:
            correct += 1
            awayWins += 1

    print('\n')
    print('Linear regression accuracy: ' + str(correct/len(homePredicted)))
    print('Percent that model pridicts away team winning: ' + str(awayWins/len(homePredicted)))
    print('\n')

    return (homePredictTrain, awayPredictTrain, homePredicted, awayPredicted)


'''
Trains and runs our SVM model
'''
def svmClassification(data, homePredicted=None, awayPredicted=None, homePredicted2=None, awayPredicted2=None):
    print('--------SVM CLASSIFICATION OUTPUT--------\n')

    training = data[0]
    yHomeTraining = data[1]
    yAwayTraining = data[2]

    validation = data[3]
    yHomeValidation = data[4]
    yAwayValidation = data[5]

    test = data[6]
    yHomeTest = data[7]
    yAwayTest = data[8]


    # Adds linear regression predicted output as features
    if homePredicted is not None and awayPredicted is not None:
        for i in range(len(homePredicted)):
            np.append(training[i], homePredicted[i])
            np.append(training[i], awayPredicted[i])

    if homePredicted2 is not None and awayPredicted2 is not None:
        for i in range(len(homePredicted2)):
            np.append(test[i], homePredicted2[i])
            np.append(test[i], awayPredicted2[i])

    classificationModel = svm.SVC(gamma='auto', C=1000.0, kernel='sigmoid')

    yClassTraining = classifyMatch(yHomeTraining, yAwayTraining)
    svmLearned = classificationModel.fit(training, yClassTraining)

    # yClassValidation = classifyMatch(yHomeValidation, yAwayValidation)

    yClassTest = classifyMatch(yHomeTest, yAwayTest)

    print('Classifcation score: ' + str(svmLearned.score(test, yClassTest)))

    # predicted = svmLearned.predict(validation)

    predicted = svmLearned.predict(test)

    awayWins = 0
    for outcome in predicted:
        if outcome == 2.0:
            awayWins += 1

    print('Percent that model pridicts away team winning: ' + str(awayWins / len(predicted)))

'''
Creates training and validation sets
'''
def preProcessing(X):
    shuffledX = random.sample(X, len(X))

    yHome = []
    yAway = []
    newX = []
    for match in shuffledX:
        yHome.append(match[0])
        yAway.append(match[1])
        newX.append(match[2:])

    # training = featureNormalize(newX[:16000])
    training = skp.normalize(newX[:16000])
    yHomeTraining = yHome[:16000]
    yAwayTraining = yAway[:16000]

    # validation = featureNormalize(newX[16001:20000])
    validation = skp.normalize(newX[16001:20000])
    yHomeValidation = yHome[16001:20000]
    yAwayValidation = yAway[16001:20000]

    test = skp.normalize(newX[20000:])
    yHomeTest = yHome[20000:]
    yAwayTest = yAway[20000:]

    return (training, yHomeTraining, yAwayTraining, validation, yHomeValidation, yAwayValidation, test, yHomeTest, yAwayTest)


teamData = readTeam_Data('Team_Attributes.csv')[0]
X = readMatch_Data('Match_Data.csv', teamData)[0]

data = preProcessing(X)

(homePredicted1, awayPredicted1, homePredicted2, awayPredicted2) = linearRegression(data)
svmClassification(data, homePredicted1, awayPredicted1, homePredicted2, awayPredicted2)
