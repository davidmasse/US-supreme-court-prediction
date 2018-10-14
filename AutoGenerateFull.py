## LIBRARIES ##
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from poibin import PoiBin
import pytest

## IMPORT ML MODELS ##
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, confusion_matrix, precision_recall_fscore_support
import xgboost as xgb


## CREATE FILE ##
#This section initializes the file, gives it a title and a timestap
now = datetime.datetime.now()
unique_report = input('Enter Unique Report Identifier - ')
fhand = open('OutcomeReport{}.txt'.format(unique_report), 'w+')

# fhand_meta = open('MetaInfoForOutcomeReport{}.csv'.format(unique_report), 'w+')
# with open('MetaInfoForOutcomeReport{}.csv'.format(unique_report), mode='w+') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',')

report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
fhand.write('SCOTUS Machine Learning Models Outcome Report {} \n\n'.format(unique_report))
fhand.write(report_time)
fhand.write('\n\n')

## DATA IMPORT & CLEANING ##
# setting random seed
np.random.seed(9)

# import data and do initial cleaning
og_data = pd.read_csv('SCDB_2018_01_justiceCentered_Citation.csv', encoding = 'ISO-8859-1')
og_data = og_data.drop(columns = ['justice', 'docketId', 'caseIssuesId', 'voteId', 'dateDecision',
                                        'usCite', 'sctCite', 'ledCite', 'lexisCite',
                                        'docket', 'caseName', 'petitionerState', 'respondentState',
                                        'adminActionState', 'caseOriginState',
                                        'caseSourceState', 'declarationUncon',
                                        'caseDispositionUnusual', 'partyWinning', 'voteUnclear',
                                        'decisionDirectionDissent', 'authorityDecision1', 'authorityDecision2',
                                        'lawSupp', 'lawMinor', 'majOpinWriter', 'majOpinAssigner',
                                        'splitVote','firstAgreement', 'secondAgreement',
                                        'dateArgument', 'dateRearg', 'petitioner', 'respondent',
                                        'term', 'caseDisposition', 'decisionDirection',
                                        'majVotes', 'minVotes', 'majority', 'vote', 'opinion',
                                        'precedentAlteration', 'issueArea'])

# drop rows null for target column, fill in other nulls, and shift targrt column so sklearn recognizes it as binary
og_data = og_data.dropna(subset=['direction'])
og_data = og_data.fillna(int(999))
d2 = {1: 0, 2: 1}
og_data['direction'] = og_data['direction'].map(d2)
for c in og_data.columns:
    og_data[c] = og_data[c].astype('category')
not_to_dummy = ['caseId', 'justiceName', 'direction']
wd_columns_to_dummy = list(og_data.columns)
for n in not_to_dummy:
    wd_columns_to_dummy.remove(n)

# save list of "features used"
fhand.write('Features_used: {} \n\n'.format(wd_columns_to_dummy))

og_data = pd.get_dummies(og_data, columns = wd_columns_to_dummy)

# splitting all case data into top-level train and test sets
full_cases = pd.read_csv('SCDB_2018_01_caseCentered_Citation.csv', encoding = 'ISO-8859-1')
full_cases = full_cases.dropna(subset=['decisionDirection'])
full_cases = full_cases.fillna(int(999))
d = {1: 0, 2: 1, 3: 3}
full_cases['decisionDirection'] = full_cases['decisionDirection'].map(d)
# the line below is not strictly necessary as the formation of justice-level training data
# generally removes rows with no value for 'direction,' and these rows are precisely the ones with
# the value 3 in the 'decisionDirection' column
full_cases = full_cases[full_cases['decisionDirection'].isin([0,1])]
cases = full_cases['caseId']
full_cases_target = full_cases['decisionDirection']
full_cases_data = full_cases[wd_columns_to_dummy]
for c in full_cases_data.columns:
    full_cases_data[c] = full_cases_data[c].astype('category')
full_cases_data = pd.get_dummies(full_cases_data)
full_cases_train, full_cases_test, full_cases_train_target, full_cases_test_target, master_train_case, master_test_case = train_test_split(full_cases_data, full_cases_target, cases)
# print("full_cases_train: ", full_cases_train.shape)
# print("full_cases_test: ", full_cases_test.shape)
# print("master_train_case: ", master_train_case.shape)
# print("master_test_case: ", master_test_case.shape)
test_outcomes = pd.DataFrame(data = full_cases_test_target.values, index = master_test_case.values)
# print("test_outcomes: ", test_outcomes.shape)

# case-centered model:
case_forest = RandomForestClassifier(n_estimators = 2000, max_depth = 12)
case_forest.fit(full_cases_train, full_cases_train_target)
case_forest_train_predict = case_forest.predict(full_cases_train)
case_forest_test_predict = case_forest.predict(full_cases_test)
case_forest_train_score = case_forest.score(full_cases_train, full_cases_train_target)
case_forest_test_score = case_forest.score(full_cases_test, full_cases_test_target)
case_forest_train_probs = case_forest.predict_proba(full_cases_train)
case_forest_test_probs = case_forest.predict_proba(full_cases_test)
case_forest_train_log_loss = log_loss(full_cases_train_target, case_forest_train_probs[:,1])
case_forest_test_log_loss = log_loss(full_cases_test_target, case_forest_test_probs[:,1])
case_forest_train_roc_auc = roc_auc_score(full_cases_train_target, case_forest_train_probs[:,1])
case_forest_test_roc_auc = roc_auc_score(full_cases_test_target, case_forest_test_probs[:,1])
print('\nCase-based train accuracy: ', case_forest_train_score)
print('\nCase-based test accuracy: ', case_forest_test_score)
print('\nCase-based train AUC: ', case_forest_train_roc_auc)
print('\nCase-based test AUC: ', case_forest_test_roc_auc)
print('\nCase-based train log-loss: ', case_forest_train_log_loss)
print('\nCase-based test log-loss: ', case_forest_test_log_loss)
case_con_matrix = confusion_matrix(full_cases_test_target, case_forest_test_predict)
print('\nCase-based test confusion Matrix:\n', case_con_matrix)
precision, recall, fscore, support = precision_recall_fscore_support(full_cases_test_target, case_forest_test_predict)
percent_conservative = support[0]/(support[0] + support[1])
print('\nBased on ', support[0], ' conservative test decsions and ', support[1], ' liberal ones (', percent_conservative, ' conservative):')
print('\nConservatism Precision: ', precision[0], '\nConservatism Recall: ', recall[0], '\nConservatism F1: ', fscore[0])
print('\nLiberalism Precision: ', precision[1], '\nConservatism Recall: ', recall[1], '\nLiberalism F1: ', fscore[1])

fhand.write("Case-based train accuracy: {}\n\n".format(case_forest_train_score))
fhand.write("Case-based test accuracy: {}\n\n".format(case_forest_test_score))
fhand.write("Case-based train AUC: {}\n\n".format(case_forest_train_roc_auc))
fhand.write("Case-based test AUC: {}\n\n".format(case_forest_test_roc_auc))
fhand.write("Case-based train log-loss: {}\n\n".format(case_forest_train_log_loss))
fhand.write("Case-based test log-loss: {}\n\n".format(case_forest_test_log_loss))
fhand.write("Case-based test confusion Matrix: {}\n\n".format(case_con_matrix))
fhand.write('Based on {} conservative test decsions and {} liberal ones ({} conservative):'.format(support[0],support[1],percent_conservative))
fhand.write('\nConservatism Precision: {}\nConservatism Recall: {}\nConservatism F1: {}'.format(precision[0],recall[0],fscore[0]))
fhand.write('\nLiberalism Precision: {}\nLiberalism Recall: {}\nLiberalism F1: {}'.format(precision[1],recall[1],fscore[1]))

## INTIALIZING JUSITCE DATA ##
justices = list(og_data.justiceName.unique())
# use shorter list below for testing purposes
# justices = ['RBGinsburg', 'AScalia', 'SAAlito']

# properly narrow data for later ensemble
working_train_data = og_data[og_data['caseId'].isin(list(master_train_case.values))]
working_test_data = og_data[og_data['caseId'].isin(list(master_test_case.values))]

# used at the end for ensemble method
master_probas = pd.DataFrame(columns = justices, index = master_test_case.values)

# Create list to hold meta_information lists for eventual DataFrame (and export)
rounds_info_master = []

## MACHINE LEARNING MODELS - CLASSIFICATION ##

fhand.write('Models will be run for {} justices\n\n'.format(len(justices)))
now = datetime.datetime.now()
report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
fhand.write(report_time)

fhand.write('Random Forest Classifier tuned on:\n')
fhand.write('n-estimators = [50, 250, 500]\n')
fhand.write('max_depth = [5, 10, 15, 20, 25, 30]\n')
fhand.write('for the roc_auc metric.\n\n')

# fhand.write('XGBoostClassifier tuned on:\n')
# fhand.write('alpha = [0.001, 0.01, 0.1, 0.2]\n')
# fhand.write('n-estimators = [100, 200, 300]\n')
# fhand.write('max_depth = [1, 2, 4, 6]\n')
# fhand.write('for the roc_auc metric.\n\n')

fhand.write('AdaBoost Classifier tuned on:\n')
fhand.write('alpha = [0.001, 0.01, 0.1]\n')
fhand.write('n-estimators = [100, 200, 300]\n')
fhand.write('max_depth = [1, 3, 6]\n')
fhand.write('for the roc_auc metric.\n\n')

fhand.write('Support Vector Machine tuned on:\n')
fhand.write('kernel = [linear, rbf, sigmoid]\n')
fhand.write('c_value = [1, 5, 10, 25, 50, 75, 100]\n\n')

# fhand.write('Logisstic Regression tuned on:\n')
# fhand.write('kernel = [linear, rbf, sigmoid]\n')
# fhand.write('c_value = [1, 5, 10, 25, 50, 75, 100]\n\n')

model_run_count = 0
print('')
print('Start Time: ')
print(report_time)

for i in range(len(justices)):

    fhand.write('************************************************')
    fhand.write("\n\n")

    current_justice = justices[model_run_count]

    model_run_count += 1

    fhand.write('Model Set {} - Justice {}\n\n'.format(model_run_count, current_justice))

    # BUILD JUSTICE DATAFRAME #
    current_justice_train_df = working_train_data[working_train_data['justiceName'] == current_justice]
    current_justice_test_df = working_test_data[working_test_data['justiceName'] == current_justice]
    case_test = current_justice_test_df['caseId']
    current_justice_train_df = current_justice_train_df.drop(columns = ['caseId', 'justiceName'])
    current_justice_test_df = current_justice_test_df.drop(columns = ['caseId', 'justiceName'])

    #pull out target vector
    current_justice_target_train = current_justice_train_df['direction']
    current_justice_data_train = current_justice_train_df.drop(columns = ['direction'])
    current_justice_target_test = current_justice_test_df['direction']
    current_justice_data_test = current_justice_test_df.drop(columns = ['direction'])

    # INTIALIZING MODELS #

    ### Random Forest ###
    forest = RandomForestClassifier(n_estimators = 100, max_depth = 15)
    forest.fit(current_justice_data_train, current_justice_target_train)

    # Initial Outcome Metrics
    forest_initial_train_score = forest.score(current_justice_data_train, current_justice_target_train)
    forest_initial_test_score = forest.score(current_justice_data_test, current_justice_target_test)

    initial_forest_train_probs = forest.predict_proba(current_justice_data_train)
    initial_forest_test_probs = forest.predict_proba(current_justice_data_test)

    initial_forest_train_predict = forest.predict(current_justice_data_train)
    initial_forest_test_predict = forest.predict(current_justice_data_test)

    forest_initial_train_log_loss = log_loss(current_justice_target_train, initial_forest_train_probs[:,1])
    forest_initial_test_log_loss = log_loss(current_justice_target_test, initial_forest_test_probs[:,1])

    forest_initial_train_roc_auc = roc_auc_score(current_justice_target_train, initial_forest_train_probs[:,1])
    forest_initial_test_roc_auc = roc_auc_score(current_justice_target_test, initial_forest_test_probs[:,1])

    # Hyperparamater Tuning

    param_grid_forest = {'n_estimators' : [50, 250, 500, 1000], 'max_depth' : [5, 10, 15, 25]}

    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(forest, param_grid_forest, scoring = "accuracy", n_jobs = -1, cv = 4)
    grid_result = grid_search.fit(current_justice_data_train, current_justice_target_train)

    # Interpreting results
    forest_best_score = grid_result.best_score_
    forest_best_params = grid_result.best_params_

    # Reintializing model with best parameters

    forest = RandomForestClassifier(max_depth = forest_best_params['max_depth'], n_estimators = forest_best_params['n_estimators'])

    forest.fit(current_justice_data_train, current_justice_target_train)

    # Final Metrics

    forest_tuned_train_score = forest.score(current_justice_data_train, current_justice_target_train)
    forest_tuned_test_score = forest.score(current_justice_data_test, current_justice_target_test)

    tuned_forest_train_probs= forest.predict_proba(current_justice_data_train)
    tuned_forest_test_probs= forest.predict_proba(current_justice_data_test)
    probs_series = pd.DataFrame(data = tuned_forest_test_probs[:,1], index = case_test.index)
    probs_with_ids = pd.concat([probs_series, case_test], axis = 1)
    probs_with_ids.rename(columns={0:'probability'}, inplace = True)
    for ind, row in probs_with_ids.iterrows():
        case = row['caseId']
        probabil = row['probability']
        master_probas[current_justice].loc[case] = probabil

    tuned_forest_train_predict = forest.predict(current_justice_data_train)
    tuned_forest_test_predict = forest.predict(current_justice_data_test)

    forest_tuned_train_log_loss = log_loss(current_justice_target_train, tuned_forest_train_probs)
    forest_tuned_test_log_loss = log_loss(current_justice_target_test, tuned_forest_test_probs)

    forest_tuned_train_roc_auc = roc_auc_score(current_justice_target_train, tuned_forest_train_probs[:,1])
    forest_tuned_test_roc_auc = roc_auc_score(current_justice_target_test, tuned_forest_test_probs[:,1])

    # Write to file

    fhand.write("Random Forest Model Results\n\n")

    now = datetime.datetime.now()
    report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    fhand.write(report_time)
    fhand.write("\n\n")

    fhand.write("On default settings, Random Forest accuracy score on training data was {}\n\n".format(forest_initial_train_score))

    fhand.write("On default settings, Random Forest accuracy score on test data was {}\n\n".format(forest_initial_test_score))

    fhand.write("On default settings, Random Forest logloss on training set was {}\n\n".format(forest_initial_train_log_loss))

    fhand.write("On default settings, Random Forest logloss on test set was {}\n\n".format(forest_initial_test_log_loss))

    fhand.write("On default settings, Random Forest roc_auc on training set was {}\n\n".format(forest_initial_train_roc_auc))

    fhand.write("On default settings, Random Forest roc_auc on test set was {}\n\n".format(forest_initial_test_roc_auc))

    fhand.write("The best roc_auc score achieved by GridSearchCV was {}\nIt was achieved by setting max depth to {} and n_estimators to {}\n\n".format(forest_best_score, forest_best_params['max_depth'], forest_best_params['n_estimators']))

    fhand.write("Once tuned, Random Forest accuracy score on training data was {}\n\n".format(forest_tuned_train_score))

    fhand.write("Once tuned, Random Forest accuracy score on test data was {}\n\n".format(forest_tuned_test_score))

    fhand.write("Once tuned, Random Forest logloss on training set was {}\n\n".format(forest_tuned_train_log_loss))

    fhand.write("Once tuned, Random Forest logloss on test set was {}\n\n".format(forest_tuned_test_log_loss))

    fhand.write("Once tuned, Random Forest roc_auc on training set was {}\n\n".format(forest_tuned_train_roc_auc))

    fhand.write("Once tuned, Random Forest roc_auc on test set was {}\n\n".format(forest_tuned_test_roc_auc))

    fhand.write('-----------------------------------------')
    fhand.write("\n\n")

## XGBoost is ready but currently not on; needs work

    # ### XGBoost ###
    #
    # # Initialize Model
    #
    # xgboost = xgb.XGBClassifier()
    # xgboost.fit(current_justice_data_train, current_justice_target_train)
    #
    # # Initial Outcome Metrics
    #
    # xgboost_initial_train_score = xgboost.score(current_justice_data_train, current_justice_target_train)
    # xgboost_initial_test_score = xgboost.score(current_justice_data_test, current_justice_target_test)
    #
    # initial_xgboost_train_probs = xgboost.predict_proba(current_justice_data_train)
    # initial_xgboost_test_probs = xgboost.predict_proba(current_justice_data_test)
    #
    # initial_xgboost_train_predict = xgboost.predict(current_justice_data_train)
    # initial_xgboost_test_predict = xgboost.predict(current_justice_data_test)
    #
    # xgboost_initial_train_log_loss = log_loss(current_justice_target_train, initial_xgboost_train_probs)
    # xgboost_initial_test_log_loss = log_loss(current_justice_target_test, initial_xgboost_test_probs)
    #
    # xgboost_initial_train_roc_auc = roc_auc_score(current_justice_target_train, initial_xgboost_train_probs[:,1])
    # xgboost_initial_test_roc_auc = roc_auc_score(current_justice_target_test, initial_xgboost_test_probs[:,1])
    #
    # # Hyperparamater Tuning
    #
    # xgboost_alpha = [ 0.01, 0.1]
    # xgboost_n_estimators = [100, 200, 300]
    # xgboost_max_depth = [1, 3, 6]
    #
    # param_grid_xgboost = dict(n_estimators=xgboost_n_estimators, max_depth=xgboost_max_depth, learning_rate = xgboost_alpha)
    #
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    #
    # grid_search = GridSearchCV(xgboost, param_grid_xgboost, scoring="roc_auc", n_jobs=-1, cv=kfold)
    # grid_result = grid_search.fit(current_justice_data_train, current_justice_target_train)
    #
    # # Interpreting results
    # xgboost_best_score = grid_result.best_score_
    # xgboost_best_params = grid_result.best_params_
    #
    # # Reintializing model with best parameters
    #
    # xgboost = xgb.XGBClassifier(max_depth = xgboost_best_params['max_depth'], n_estimators = xgboost_best_params['n_estimators'], learning_rate = xgboost_best_params['learning_rate'])
    #
    # xgboost.fit(current_justice_data_train, current_justice_target_train)
    #
    # # Final Metrics
    #
    # xgboost_tuned_train_score = xgboost.score(current_justice_data_train, current_justice_target_train)
    # xgboost_tuned_test_score = xgboost.score(current_justice_data_test, current_justice_target_test)
    #
    # tuned_xgboost_train_probs= xgboost.predict_proba(current_justice_data_train)
    # tuned_xgboost_test_probs= xgboost.predict_proba(current_justice_data_test)
    #
    # tuned_xgboost_train_predict = xgboost.predict(current_justice_data_train)
    # tuned_xgboost_test_predict = xgboost.predict(current_justice_data_test)
    #
    # xgboost_tuned_train_log_loss = log_loss(current_justice_target_train, tuned_xgboost_train_probs)
    # xgboost_tuned_test_log_loss = log_loss(current_justice_target_test, tuned_xgboost_test_probs)
    #
    # xgboost_tuned_train_roc_auc = roc_auc_score(current_justice_target_train, tuned_xgboost_train_probs[:,1])
    # xgboost_tuned_test_roc_auc = roc_auc_score(current_justice_target_test, tuned_xgboost_test_probs[:,1])
    #
    # # Write to file
    #
    # fhand.write("XGBoost Model Results\n\n")
    #
    # now = datetime.datetime.now()
    # report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    # fhand.write(report_time)
    # fhand.write("\n\n")
    #
    # fhand.write("On default settings, XGBoost accuracy score on training data was {}\n\n".format(xgboost_initial_train_score))
    #
    # fhand.write("On default settings, XGBoost accuracy score on test data was {}\n\n".format(xgboost_initial_test_score))
    #
    # fhand.write("On default settings, XGBoost logloss on training set was {}\n\n".format(xgboost_initial_train_log_loss))
    #
    # fhand.write("On default settings, XGBoost logloss on test set was {}\n\n".format(xgboost_initial_test_log_loss))
    #
    # fhand.write("On default settings, XGBoost roc_auc on training set was {}\n\n".format(xgboost_initial_train_roc_auc))
    #
    # fhand.write("On default settings, XGBoost roc_auc on test set was {}\n\n".format(xgboost_initial_test_roc_auc))
    #
    # fhand.write("The best roc_auc score achieved by GridSearchCV was {}\nIt was achieved by setting alpha to {}, max depth to {}, and n_estimators to {}\n\n".format(xgboost_best_score, xgboost_best_params['learning_rate'], xgboost_best_params['max_depth'], xgboost_best_params['n_estimators']))
    #
    # fhand.write("Once tuned, XGBoost accuracy score on training data was {}\n\n".format(xgboost_tuned_train_score))
    #
    # fhand.write("Once tuned, XGBoost accuracy score on test data was {}\n\n".format(xgboost_tuned_test_score))
    #
    # fhand.write("Once tuned, XGBoost logloss on training set was {}\n\n".format(xgboost_tuned_train_log_loss))
    #
    # fhand.write("Once tuned, XGBoost logloss on test set was {}\n\n".format(xgboost_tuned_test_log_loss))
    #
    # fhand.write("Once tuned, XGBoost roc_auc on training set was {}\n\n".format(xgboost_tuned_train_roc_auc))
    #
    # fhand.write("Once tuned, XGBoost roc_auc on test set was {}\n\n".format(xgboost_tuned_test_roc_auc))
    #
    # fhand.write('-----------------------------------------')
    # fhand.write("\n\n")



    ### AdaBoost ###

    # Initialize Model

    adaboost = AdaBoostClassifier()
    adaboost.fit(current_justice_data_train, current_justice_target_train)

    # Initial Outcome Metrics

    adaboost_initial_train_score = adaboost.score(current_justice_data_train, current_justice_target_train)
    adaboost_initial_test_score = adaboost.score(current_justice_data_test, current_justice_target_test)

    initial_adaboost_train_probs = adaboost.predict_proba(current_justice_data_train)
    initial_adaboost_test_probs = adaboost.predict_proba(current_justice_data_test)

    initial_adaboost_train_predict = adaboost.predict(current_justice_data_train)
    initial_adaboost_test_predict = adaboost.predict(current_justice_data_test)

    adaboost_initial_train_log_loss = log_loss(current_justice_target_train, initial_adaboost_train_probs)
    adaboost_initial_test_log_loss = log_loss(current_justice_target_test, initial_adaboost_test_probs)

    adaboost_initial_train_roc_auc = roc_auc_score(current_justice_target_train, initial_adaboost_train_probs[:,1])
    adaboost_initial_test_roc_auc = roc_auc_score(current_justice_target_test, initial_adaboost_test_probs[:,1])

    # Hyperparamater Tuning

    param_grid_adaboost = {'learning_rate' : [0.001, 0.01, 0.1], 'n_estimators' : [100, 300, 600]}

    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(adaboost, param_grid_adaboost, scoring="roc_auc", n_jobs=-1, cv=4)
    grid_result = grid_search.fit(current_justice_data_train, current_justice_target_train)

    # Interpreting results
    adaboost_best_score = grid_result.best_score_
    adaboost_best_params = grid_result.best_params_

    # Reintializing model with best parameters

    adaboost = AdaBoostClassifier(n_estimators = adaboost_best_params['n_estimators'], learning_rate = adaboost_best_params['learning_rate'])

    adaboost.fit(current_justice_data_train, current_justice_target_train)

    # Final Metrics

    adaboost_tuned_train_score = adaboost.score(current_justice_data_train, current_justice_target_train)
    adaboost_tuned_test_score = adaboost.score(current_justice_data_test, current_justice_target_test)

    tuned_adaboost_train_probs= adaboost.predict_proba(current_justice_data_train)
    tuned_adaboost_test_probs= adaboost.predict_proba(current_justice_data_test)

    tuned_adaboost_train_predict = adaboost.predict(current_justice_data_train)
    tuned_adaboost_test_predict = adaboost.predict(current_justice_data_test)

    adaboost_tuned_train_log_loss = log_loss(current_justice_target_train, tuned_adaboost_train_probs)
    adaboost_tuned_test_log_loss = log_loss(current_justice_target_test, tuned_adaboost_test_probs)

    adaboost_tuned_train_roc_auc = roc_auc_score(current_justice_target_train, tuned_adaboost_train_probs[:,1])
    adaboost_tuned_test_roc_auc = roc_auc_score(current_justice_target_test, tuned_adaboost_test_probs[:,1])

    # Write to file

    fhand.write("AdaBoost Model Results\n\n")

    now = datetime.datetime.now()
    report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    fhand.write(report_time)
    fhand.write("\n\n")

    fhand.write("On default settings, AdaBoost accuracy score on training data was {}\n\n".format(adaboost_initial_train_score))

    fhand.write("On default settings, AdaBoost accuracy score on test data was {}\n\n".format(adaboost_initial_test_score))

    fhand.write("On default settings, AdaBoost logloss on training set was {}\n\n".format(adaboost_initial_train_log_loss))

    fhand.write("On default settings, AdaBoost logloss on test set was {}\n\n".format(adaboost_initial_test_log_loss))

    fhand.write("On default settings, AdaBoost roc_auc on training set was {}\n\n".format(adaboost_initial_train_roc_auc))

    fhand.write("On default settings, AdaBoost roc_auc on test set was {}\n\n".format(adaboost_initial_test_roc_auc))

    fhand.write("The best roc_auc score achieved by GridSearchCV was {}\nIt was achieved by setting learning_rate to {} and n_estimators to {}\n\n".format(adaboost_best_score, adaboost_best_params['learning_rate'], adaboost_best_params['n_estimators']))

    fhand.write("Once tuned, AdaBoost accuracy score on training data was {}\n\n".format(adaboost_tuned_train_score))

    fhand.write("Once tuned, AdaBoost accuracy score on test data was {}\n\n".format(adaboost_tuned_test_score))

    fhand.write("Once tuned, AdaBoost logloss on training set was {}\n\n".format(adaboost_tuned_train_log_loss))

    fhand.write("Once tuned, AdaBoost logloss on test set was {}\n\n".format(adaboost_tuned_test_log_loss))

    fhand.write("Once tuned, AdaBoost roc_auc on training set was {}\n\n".format(adaboost_tuned_train_roc_auc))

    fhand.write("Once tuned, AdaBoost roc_auc on test set was {}\n\n".format(adaboost_tuned_test_roc_auc))

    fhand.write('-----------------------------------------')
    fhand.write("\n\n")

    ### Support Vector Machine ###

    # Intialize Model

    svm_model = svm.SVC(probability=True, random_state=7)
    svm_model.fit(current_justice_data_train, current_justice_target_train)

    # Hyperparamater Tuning

    svm_model_kernel = ["linear", "rbf"]
    svm_model_C_value = [1, 5, 10, 25, 50, 75, 100]

    param_grid_svm_model = dict(C = svm_model_C_value, kernel = svm_model_kernel)

    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    grid_search = GridSearchCV(svm_model, param_grid_svm_model, scoring="roc_auc",  n_jobs=-1, cv=4)
    grid_result = grid_search.fit(current_justice_data_train, current_justice_target_train)

    # Interpreting results
    svm_model_best_score = grid_result.best_score_
    svm_model_best_params = grid_result.best_params_

    # Reintializing model with best parameters

    svm_model = svm.SVC(C = svm_model_best_params['C'], kernel = svm_model_best_params['kernel'], probability=True, random_state=7)

    svm_model.fit(current_justice_data_train, current_justice_target_train)

    # Final Metrics

    svm_model_tuned_train_score = svm_model.score(current_justice_data_train, current_justice_target_train)
    svm_model_tuned_test_score = svm_model.score(current_justice_data_test, current_justice_target_test)

    tuned_svm_model_train_probs= svm_model.predict_proba(current_justice_data_train)
    tuned_svm_model_test_probs= svm_model.predict_proba(current_justice_data_test)

    svm_model_tuned_train_log_loss = log_loss(current_justice_target_train, tuned_svm_model_train_probs)
    svm_model_tuned_test_log_loss = log_loss(current_justice_target_test, tuned_svm_model_test_probs)

    svm_model_tuned_train_roc_auc = roc_auc_score(current_justice_target_train, tuned_svm_model_train_probs[:,1])
    svm_model_tuned_test_roc_auc = roc_auc_score(current_justice_target_test, tuned_svm_model_test_probs[:,1])


    fhand.write("Support Vector Machine Results\n\n")

    now = datetime.datetime.now()
    report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    fhand.write(report_time)
    fhand.write("\n\n")

    fhand.write("The best roc_auc score achieved by GridSearchCV was {}\nIt was achieved by setting C to {} and kernel type to {}\n\n".format(svm_model_best_score, svm_model_best_params['C'], svm_model_best_params['kernel']))

    fhand.write("Once tuned, SVM accuracy score on training data was {}\n\n".format(svm_model_tuned_train_score))

    fhand.write("Once tuned, SVM accuracy score on test data was {}\n\n".format(svm_model_tuned_test_score))

    fhand.write("Once tuned, SVM logloss on training set was {}\n\n".format(svm_model_tuned_train_log_loss))

    fhand.write("Once tuned, SVM logloss on test set was {}\n\n".format(svm_model_tuned_test_log_loss))

    fhand.write("Once tuned, SVM roc_auc on training set was {}\n\n".format(svm_model_tuned_train_roc_auc))

    fhand.write("Once tuned, SVM roc_auc on test set was {}\n\n".format(svm_model_tuned_test_roc_auc))

    fhand.write('-----------------------------------------')
    fhand.write("\n\n")


    ### Logistic Regression ###

    #
    # fhand.write("Logistic Regression Results\n\n")
    #
    # now = datetime.datetime.now()
    # report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    # fhand.write(report_time)
    # fhand.write("\n\n")


    # SAVE JUSTICE SPECIFIC OUTCOMES #

    # write meta analysis to a csv
    round_info = [current_justice, model_run_count, forest_initial_train_score, forest_initial_test_score, forest_initial_train_log_loss, forest_initial_test_log_loss, forest_initial_train_roc_auc, forest_initial_test_roc_auc, forest_best_score, forest_best_params['max_depth'], forest_best_params['n_estimators'], forest_tuned_train_score, forest_tuned_test_score, forest_tuned_train_log_loss, forest_tuned_test_log_loss, forest_tuned_train_roc_auc, forest_tuned_test_roc_auc, adaboost_initial_train_score, adaboost_initial_test_score, adaboost_initial_train_log_loss, adaboost_initial_test_log_loss, adaboost_initial_train_roc_auc, adaboost_initial_test_roc_auc, adaboost_best_score, adaboost_best_params['learning_rate'], adaboost_best_params['n_estimators'],
    adaboost_tuned_train_score, adaboost_tuned_test_score, adaboost_tuned_train_log_loss, adaboost_tuned_test_log_loss, adaboost_tuned_train_roc_auc, adaboost_tuned_test_roc_auc, svm_model_best_score, svm_model_best_params['C'], svm_model_best_params['kernel'], svm_model_tuned_train_score, svm_model_tuned_test_score, svm_model_tuned_train_log_loss, svm_model_tuned_test_log_loss, svm_model_tuned_train_roc_auc, svm_model_tuned_test_roc_auc]

    rounds_info_master.append(round_info)

#xgboost meta tags

# xgboost_initial_train_score, xgboost_initial_test_score, xgboost_initial_train_log_loss, xgboost_initial_test_log_loss, xgboost_initial_train_roc_auc, xgboost_initial_test_roc_auc, xgboost_best_score, xgboost_best_params['learning_rate'], xgboost_best_params['max_depth'],
# xgboost_best_params['n_estimators'], xgboost_tuned_train_score, xgboost_tuned_test_score, xgboost_tuned_train_log_loss, xgboost_tuned_test_log_loss, xgboost_tuned_train_roc_auc, xgboost_tuned_test_roc_auc,



    #create lists of feature coefficients, then add id column info

    model_type = "RFT"
    forest_feature_import = list(forest.feature_importances_)
    forest_feature_import_with_id = ['{}-{}-{}'.format(current_justice, model_type, model_run_count)] + forest_feature_import

    # model_type = "XGB"
    # xgb_feature_import = list(xgboost.feature_importances_)
    # xgb_feature_import_with_id = ['{}-{}-{}'.format(current_justice, model_type, model_run_count)] + xgb_feature_import

    model_type = "ADA"
    ada_feature_import = list(adaboost.feature_importances_)
    ada_feature_import_with_id = ['{}-{}-{}'.format(current_justice, model_type, model_run_count)] + ada_feature_import

    # model_type = "SVM"
    # svm_feature_import = list(svm_model.feature_importances_)
    # svm_feature_import_with_id = ['{}-{}-{}'.format(current_justice, model_type, model_run_count)] + svm_feature_import

    # model_type = "LGR"
    # lgr_feature_import = list(logreg.feature_importances_)
    # lgr_feature_import_with_id = ['{}-{}-{}'.format(current_justice, model_type, model_run_count)] + lgr_feature_import


    feature_info_master = []

    feature_columns_info_master =[]

    feature_info_master.append(forest_feature_import_with_id)
    # feature_info_master.append(xgb_feature_import_with_id)
    feature_info_master.append(ada_feature_import_with_id)
    # feature_info_master.append(svm_feature_import_with_id)
    # feature_info_master.append(lgr_feature_import_with_id)

    features_as_a_list = list(current_justice_data_train.columns)
    features_master_columns = ['ID'] + features_as_a_list
    feature_columns_info_master.append(features_master_columns)

    feature_master = pd.DataFrame.from_records(feature_info_master, columns = feature_columns_info_master)
    feature_master.to_csv('OutcomeReport_{}_FeatureImportInfo{}.csv'.format(unique_report, current_justice), mode = 'w+')

    print("Round {} - Justice {} Done.\n".format(model_run_count, current_justice))

    now = datetime.datetime.now()
    report_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
    print("At:" + report_time)

    # fhand.close()



#Creating meta_information DataFrame for export to csv

meta_master = pd.DataFrame.from_records(rounds_info_master, columns = ['JusticeName', 'Round',
                    'forest_initial_train_score', 'forest_initial_test_score', 'forest_initial_train_log_loss', 'forest_initial_test_log_loss', 'forest_initial_train_roc_auc', 'forest_initial_test_roc_auc', 'forest_best_score', 'forest_best_max_depth' , 'forest_best_n_estimators', 'forest_tuned_train_score', 'forest_tuned_test_score', 'forest_tuned_train_log_loss', 'forest_tuned_test_log_loss', 'forest_tuned_train_roc_auc', 'forest_tuned_test_roc_auc', 'adaboost_initial_train_score', 'adaboost_initial_test_score', 'adaboost_initial_train_log_loss', 'adaboost_initial_test_log_loss', 'adaboost_initial_train_roc_auc',
                    'adaboost_initial_test_roc_auc', 'adaboost_best_score', 'adaboost_best_learning_rate', 'adaboost_best_n_estimators', 'adaboost_tuned_train_score', 'adaboost_tuned_test_score', 'adaboost_tuned_train_log_loss', 'adaboost_tuned_test_log_loss', 'adaboost_tuned_train_roc_auc', 'adaboost_tuned_test_roc_auc', 'svm_model_best_score', 'svm_model_best_c', 'svm_model_best_kernel', 'svm_model_tuned_train_score', 'svm_model_tuned_test_score', 'svm_model_tuned_train_log_loss', 'svm_model_tuned_test_log_loss', 'svm_model_tuned_train_roc_auc', 'svm_model_tuned_test_roc_auc'])


#xgboost meta column tags

# 'xgboost_initial_test_score', 'xgboost_initial_train_log_loss', 'xgboost_initial_test_log_loss', 'xgboost_initial_train_roc_auc',
# 'xgboost_initial_test_roc_auc', 'xgboost_best_score', 'xgboost_best_alpha', 'xgboost_best_max_depth', 'xgboost_best_n_estimators', 'xgboost_tuned_train_score', 'xgboost_tuned_test_score', 'xgboost_tuned_train_log_loss', 'xgboost_tuned_test_log_loss', 'xgboost_tuned_train_roc_auc', 'xgboost_tuned_test_roc_auc',

meta_master.to_csv('OutcomeReport{}MetaInfo.csv'.format(unique_report), mode = 'w+')

# feature_master.to_csv()


for i in range(len(justices)):
    feature_master = pd.DataFrame.from_records(feature_info_master, columns = feature_columns_info_master)
    feature_master.to_csv('OutcomeReport_{}_FeatureImportInfo{}.csv'.format(unique_report, current_justice), mode = 'w+')


master_probas = master_probas.fillna(2)
ps = dict.fromkeys(list(master_probas.index.values), 0)
for ind, row in master_probas.iterrows():
    lista = []
    for c in master_probas.columns:
        if row[c] != 2:
            lista.append(row[c])
    ps[ind] = lista
outcomes = {}
for k in ps.keys():
    pb = PoiBin(ps[k])
    if len(ps[k]) == 9:
        outcomes[k] = sum(pb.pmf([5, 6, 7, 8, 9]))
    elif len(ps[k]) == 8:
        outcomes[k] = sum(pb.pmf([5, 6, 7, 8]))
    elif len(ps[k]) == 7:
        outcomes[k] = sum(pb.pmf([4, 5, 6, 7]))
    elif len(ps[k]) == 6:
        outcomes[k] = sum(pb.pmf([4, 5, 6]))
    elif len(ps[k]) == 5:
        outcomes[k] = sum(pb.pmf([3, 4, 5]))
    elif len(ps[k]) == 4:
        outcomes[k] = sum(pb.pmf([3, 4]))
    elif len(ps[k]) == 3:
        outcomes[k] = sum(pb.pmf([2, 3]))
    elif len(ps[k]) == 2:
        outcomes[k] = sum(pb.pmf([2]))
# as it happens, the minimum number of justices to vote in a case is 5

# print("\n\nOutcomes: ", outcomes)

probs = []
case_outcomes = []
for k,v in outcomes.items():
    probs.append(v)
    case_outcomes.append(test_outcomes.loc[k])

predicted = []
for prob in probs:
    if prob > 0.5:
        val = 1
    else:
        val = 0
    predicted.append(val)

ensemble_acc = accuracy_score(case_outcomes, predicted)
print("\nJustice-based test accuracy: ", ensemble_acc)

ensemble_auc = roc_auc_score(case_outcomes, probs)
print("\nJustice-based test AUC: ", ensemble_auc)

probs2 = np.array(probs)
case_outcomes2 = np.array(case_outcomes)
ensemble_ll = log_loss(case_outcomes2, probs2)
print("\nJustice-based test log-loss: ", ensemble_ll)

cnf_matrix = confusion_matrix(case_outcomes, predicted)
print('\nJustice-based test confusion Matrix:\n',cnf_matrix)

precision, recall, fscore, support = precision_recall_fscore_support(case_outcomes, predicted)
percent_conservative = support[0]/(support[0] + support[1])
print('\nBased on ', support[0], ' conservative test decsions and ', support[1], ' liberal ones (', percent_conservative, ' conservative):')
print('\nConservatism Precision: ', precision[0], '\nConservatism Recall: ', recall[0], '\nConservatism F1: ', fscore[0])
print('\nLiberalism Precision: ', precision[1], '\nConservatism Recall: ', recall[1], '\nLiberalism F1: ', fscore[1])

finish_time = now.strftime("%m-%d-%Y %I:%M:%S %p")
print('\n\n', finish_time)

fhand.write("Justice-based test accuracy: {}\n\n".format(ensemble_acc))
fhand.write("Justice-based test AUC: {}\n\n".format(ensemble_auc))
fhand.write("Justice-based test log-loss: {}\n\n".format(ensemble_ll))
fhand.write("Justice-based test confusion Matrix: {}\n\n".format(cnf_matrix))
fhand.write('Based on {} conservative test decsions and {} liberal ones ({} conservative):'.format(support[0],support[1],percent_conservative))
fhand.write('\nConservatism Precision: {}\nConservatism Recall: {}\nConservatism F1: {}'.format(precision[0],recall[0],fscore[0]))
fhand.write('\nLiberalism Precision: {}\nLiberalism Recall: {}\nLiberalism F1: {}'.format(precision[1],recall[1],fscore[1]))

fhand.close()



## MACHINE LEARNING MODELS - REGRESSION ##
# Using Miller-Quinn scores with accuracy data of justices
