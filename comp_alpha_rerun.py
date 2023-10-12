#!/usr/bin/python3

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import *
import dill as pkl
import pandas as pd
from sklearn.metrics import mean_squared_error as loss_fn
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pdl    
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import copy
from scipy.spatial import KDTree
from sklearn.metrics import zero_one_loss
import os
import csv
import matplotlib.pyplot as plt
import definitions as defn
import importlib
import tqdm
import warnings
import time
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import *
import dill as pkl
import pandas as pd
import os
import sys
import dis
from io import StringIO # Python3 use: from io import StringIO
import sys
import tqdm
import multiprocessing
import threading
from func_timeout import func_timeout, FunctionTimedOut
multiprocessing.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")

alpha = 100_000

def main():
    groups_folder = "dataset_pairs/groups"
    hypotheses_folder = "dataset_pairs/hypotheses"

    x_train = pd.read_csv('data/training_data.csv') 
    y_train = np.genfromtxt('data/training_labels.csv', delimiter=',', dtype = float)
    x_val = pd.read_csv('data/validation_data.csv') 
    y_val = np.genfromtxt('data/validation_labels.csv', delimiter=',', dtype = float)
    x_test = pd.read_csv('data/test_data.csv') 
    y_test = np.genfromtxt('data/test_labels.csv', delimiter=',', dtype = float)

    num_pairs = len(os.listdir(f"{groups_folder}"))

    log_df = pd.read_csv("FINAL_PAPER.csv", index_col="Unnamed: 0")

    with open("initial_model.pkl", "rb") as file:
        global_clf = pkl.load(file)
        
    clf = DecisionTreeRegressor(max_depth = 1, random_state = 42)
    clf.fit(x_train, y_train)

    global_pdl = pdl.PointerDecisionList(global_clf, x_train, y_train, x_val, y_val, alpha = alpha, min_group_size = 1)
    team_pdl = pdl.PointerDecisionList(clf, x_train, y_train, x_val, y_val, alpha = alpha, min_group_size = 1)

    try:
        os.mkdir(f"alpha_{alpha}")
        os.mkdir(f"alpha_{alpha}/teams")
        os.mkdir(f"alpha_{alpha}/teams/global_pdl")
    except:
        print("Folder already exists, please check alpha!")
        exit()

    # initialize teams to have a folder
    for team in range(46):
        os.mkdir(f"alpha_{alpha}/teams/{team}")
    for team in range(46):  
        team_pdl.save_model(f"alpha_{alpha}/teams/{team}/PDL")
    global_pdl.save_model(f"alpha_{alpha}/teams/global_pdl/PDL")

    team_train_predictions = team_pdl.predict(x_train)
    team_val_predictions = team_pdl.predict(x_val)
    team_test_predictions = team_pdl.predict(x_test)

    global_train_predictions = global_pdl.predict(x_train)
    global_val_predictions = global_pdl.predict(x_val)
    global_test_predictions = global_pdl.predict(x_test)

    for team in range(46):
        np.save(f"alpha_{alpha}/teams/{team}/train_predictions", team_train_predictions)
        np.save(f"alpha_{alpha}/teams/{team}/val_predictions", team_val_predictions)
        np.save(f"alpha_{alpha}/teams/{team}/test_predictions", team_test_predictions)

        np.save(f"alpha_{alpha}/teams/global_pdl/train_predictions",global_train_predictions)
        np.save(f"alpha_{alpha}/teams/global_pdl/val_predictions", global_val_predictions)
        np.save(f"alpha_{alpha}/teams/global_pdl/test_predictions", global_test_predictions)

    information_df = pd.read_csv("intermediate_csvs/information.csv")

    columns_to_fill = set(log_df.columns) - set(information_df.columns)

    for column in columns_to_fill:
        if column in ['GGrETe', 'GGrETr', 'GGrEVa', 'TGrETe', 'TGrETr', 'TGrEVa', 'GrWeTe']:
            continue
        information_df[column] = np.nan
        for column in ['GGrEPrTe', 'GGrEPrTr', 'GGrEPrVa', 'TGrEPrTe', 'TGrEPrTr', 'TGrEPrVa', 'GGrEPoTe', 'GGrEPoTr', 'GGrEPoVa', 'TGrEPoTe', 'TGrEPoTr', 'TGrEPoVa']:
            information_df[column] = np.nan


    n_train = len(x_train)
    n_val = len(x_val)
    n_test = len(x_test)

    for i in tqdm.tqdm( range(num_pairs-1)):

        if np.isnan(information_df['TID'][i]):
            continue

        team_name = int(information_df['TID'][i])
        attempt_num = int(information_df["GAN"][i])

        team_train_predictions = np.load(f"alpha_{alpha}/teams/{team_name}/train_predictions.npy")
        team_val_predictions = np.load(f"alpha_{alpha}/teams/{team_name}/val_predictions.npy")
        team_test_predictions = np.load(f"alpha_{alpha}/teams/{team_name}/test_predictions.npy")

        global_train_predictions = np.load(f"alpha_{alpha}/teams/global_pdl/train_predictions.npy")
        global_val_predictions = np.load(f"alpha_{alpha}/teams/global_pdl/val_predictions.npy")
        global_test_predictions = np.load(f"alpha_{alpha}/teams/global_pdl/test_predictions.npy")

        with open(f"{groups_folder}/g{i+1}.pkl", "rb") as file:
            g = pkl.load(file)
            if type(information_df['GrCl'][i]) == float:
                continue
            if "Forest" not in information_df["GrCl"][i]:
                try:
                    g.__self__.n_jobs = -1
                    g.__self__.verbose = False
                except:
                    x = 1
            try:
                if "Forest" in str(g):
                    g.__self__.n_jobs = 1
            except:
                x = 1
        with open(f"{hypotheses_folder}/h{i+1}.pkl", "rb") as file:
            h = pkl.load(file)
            try:
                h.__self__.n_jobs = -1
                h.__self__.verbose = False
            except:
                x = 1
            try:
                if "Forest" in str(g):
                    g.__self__.n_jobs = 1
            except:
                x = 1
        try:
            train_indices = func_timeout(60,g, [x_train])
        except:
            continue

        val_indices = g(x_val)
        test_indices = g(x_test)

        if train_indices.dtype == float:
            continue

        group_weight_train = train_indices.sum()/n_train
        group_weight_val = val_indices.sum()/n_val
        group_weight_test = test_indices.sum()/n_test

        if ((group_weight_train > 1) or (group_weight_train == 0) or (group_weight_val > 1) or (group_weight_val == 0) or (group_weight_test > 1) or (group_weight_test == 0)):
            continue

        # h predictions
        try:
            h_train_predictions = func_timeout(60,h, [x_train])
        except:
            continue

        h_val_predictions = h(x_val)
        h_test_predictions = h(x_test)

        # h dataset error
        h_train_error = loss_fn(y_train, h_train_predictions)
        h_val_error = loss_fn(y_val, h_val_predictions)
        h_test_error = loss_fn(y_test, h_test_predictions)

        # h group error
        h_group_train_error = loss_fn(y_train[train_indices], h_train_predictions[train_indices])
        h_group_val_error = loss_fn(y_val[val_indices], h_val_predictions[val_indices])
        h_group_test_error = loss_fn(y_test[test_indices], h_test_predictions[test_indices])

        # team PDL dataset error
        pre_team_train_error = loss_fn(y_train, team_train_predictions)
        pre_team_val_error = loss_fn(y_val, team_val_predictions)
        pre_team_test_error = loss_fn(y_test, team_test_predictions)

        # team PDL group error
        pre_team_group_train_error = loss_fn(y_train[train_indices], team_train_predictions[train_indices])
        pre_team_group_val_error = loss_fn(y_val[val_indices], team_val_predictions[val_indices])
        pre_team_group_test_error = loss_fn(y_test[test_indices], team_test_predictions[test_indices])
        
        # compute delta error on g from h and team pdl
        delta_team_group_train_error = pre_team_group_train_error - h_group_train_error
        delta_team_group_val_error = pre_team_group_val_error - h_group_val_error
        delta_team_group_test_error = pre_team_group_test_error - h_group_test_error

        if pre_team_group_val_error > h_group_val_error:
            team_pdl = pdl.PointerDecisionList(clf, x_train, y_train, x_val, y_val, alpha = 100000, min_group_size = 1)
            team_pdl.reload_model(f"alpha_{alpha}/teams/{team_name}/PDL")
            team_flag = team_pdl.update(g, h, x_train, y_train, x_val, y_val)
            try:
                team_repairs = team_pdl.repairs
            except:
                team_repairs = 0
            team_updates = team_pdl.updates
        else:
            team_flag = False
            team_repairs = 0
            team_updates = 0
            
        if team_flag:
            # update predictions
            team_train_predictions = team_pdl.train_predictions
            team_val_predictions = team_pdl.val_predictions
            team_test_predictions = team_pdl.predict(x_test)

            # save new predictions
            np.save(f"alpha_{alpha}/teams/{team_name}/train_predictions.npy", team_train_predictions)
            np.save(f"alpha_{alpha}/teams/{team_name}/val_predictions.npy", team_val_predictions)
            np.save(f"alpha_{alpha}/teams/{team_name}/test_predictions.npy", team_test_predictions)
            
            team_pdl.save_model(f"alpha_{alpha}/teams/{team_name}/PDL")
            
        # compute new PDL error
        post_team_train_error = loss_fn(y_train, team_train_predictions)
        post_team_val_error = loss_fn(y_val, team_val_predictions)
        post_team_test_error = loss_fn(y_test, team_test_predictions)

        # compute PDL error change
        delta_team_train_error = pre_team_train_error - post_team_train_error
        delta_team_val_error = pre_team_val_error - post_team_val_error 
        delta_team_test_error = pre_team_test_error - post_team_test_error
            
        # team PDL post group error
        post_team_group_train_error = loss_fn(y_train[train_indices], team_train_predictions[train_indices])
        post_team_group_val_error = loss_fn(y_val[val_indices], team_val_predictions[val_indices])
        post_team_group_test_error = loss_fn(y_test[test_indices], team_test_predictions[test_indices])

        # team PDL dataset error
        pre_global_train_error = loss_fn(y_train, global_train_predictions)
        pre_global_val_error = loss_fn(y_val, global_val_predictions)
        pre_global_test_error = loss_fn(y_test, global_test_predictions)

        # global PDL group error
        pre_global_group_train_error = loss_fn(y_train[train_indices], global_train_predictions[train_indices])
        pre_global_group_val_error = loss_fn(y_val[val_indices], global_val_predictions[val_indices])
        pre_global_group_test_error = loss_fn(y_test[test_indices], global_test_predictions[test_indices])      
        
        # compute delta error on g from h and team pdl
        delta_global_group_train_error = pre_global_group_train_error - h_group_train_error
        delta_global_group_val_error = pre_global_group_val_error - h_group_val_error
        delta_global_group_test_error = pre_global_group_test_error - h_group_test_error

        if pre_global_group_val_error > h_group_val_error:
            global_pdl = pdl.PointerDecisionList(clf, x_train, y_train, x_val, y_val, alpha = 100000, min_group_size = 1)
            global_pdl.reload_model(f"alpha_{alpha}/teams/global_pdl/PDL")
            global_flag = global_pdl.update(g, h, x_train, y_train, x_val, y_val)
            try:
                global_repairs = global_pdl.repairs
            except:
                global_repairs = 0
            global_updates = global_pdl.updates
        else:
            global_flag = False
            global_repairs = 0
            global_updates = 0

        if global_flag:
            # update predictions
            global_train_predictions = global_pdl.train_predictions
            global_val_predictions = global_pdl.val_predictions
            global_test_predictions = global_pdl.predict(x_test)

            # save new predictions
            np.save(f"alpha_{alpha}/teams/global_pdl/train_predictions.npy", global_train_predictions)
            np.save(f"alpha_{alpha}/teams/global_pdl/val_predictions.npy", global_val_predictions)
            np.save(f"alpha_{alpha}/teams/global_pdl/test_predictions.npy", global_test_predictions)

            global_pdl.save_model(f"alpha_{alpha}/teams/global_pdl/PDL")

        # compute PDL error change
        post_global_train_error = loss_fn(y_train, global_train_predictions)
        post_global_val_error = loss_fn(y_val, global_val_predictions)
        post_global_test_error = loss_fn(y_test, global_test_predictions)

        # global PDL group error
        post_global_group_train_error = loss_fn(y_train[train_indices], global_train_predictions[train_indices])
        post_global_group_val_error = loss_fn(y_val[val_indices], global_val_predictions[val_indices])
        post_global_group_test_error = loss_fn(y_test[test_indices], global_test_predictions[test_indices])      

        delta_global_train_error = pre_global_train_error - post_global_train_error
        delta_global_val_error = pre_global_val_error - post_global_val_error
        delta_global_test_error = pre_global_test_error - post_global_test_error
        

        information_df['GDTe'][i] = delta_global_test_error
        information_df['GDTr'][i] = delta_global_train_error
        information_df['GDVa'][i] = delta_global_val_error

        information_df['GEPoTe'][i] = post_global_test_error
        information_df['GEPoTr'][i] = post_global_train_error
        information_df['GEPoVa'][i] = post_global_val_error

        information_df['GEPrTe'][i] = pre_global_test_error
        information_df['GEPrTr'][i] = pre_global_train_error
        information_df['GEPrVa'][i] = pre_global_val_error

        information_df['GF'][i] = global_flag

        information_df['GGrDTe'][i] = delta_global_group_test_error
        information_df['GGrDTr'][i] = delta_global_group_train_error
        information_df['GGrDVa'][i] = delta_global_group_val_error

        information_df['GGrEPrTe'][i] = pre_global_group_test_error
        information_df['GGrEPrTr'][i] = pre_global_group_train_error
        information_df['GGrEPrVa'][i] = pre_global_group_val_error


        information_df['GGrEPoTe'][i] = post_global_group_test_error
        information_df['GGrEPoTr'][i] = post_global_group_train_error
        information_df['GGrEPoVa'][i] = post_global_group_val_error


        information_df['GR'][i] = global_pdl.repairs

        information_df['GRDTe'][i] = global_flag*(delta_global_test_error - group_weight_test*delta_global_group_test_error)
        information_df['GRDTr'][i] = global_flag*(delta_global_train_error - group_weight_train*delta_global_group_train_error)
        information_df['GRDVa'][i] = global_flag*(delta_global_val_error - group_weight_val*delta_global_group_val_error)

        information_df['GU'][i] = global_pdl.updates

        information_df['GrWTe'][i] = group_weight_test
        information_df['GrWTr'][i] = group_weight_train
        information_df['GrWVa'][i] = group_weight_val

        information_df['HETe'][i] = h_test_error
        information_df['HETr'][i] = h_train_error
        information_df['HEVa'][i] = h_val_error

        information_df['HGrETe'][i] = h_group_test_error
        information_df['HGrETr'][i] = h_group_train_error
        information_df['HGrEVa'][i] = h_group_val_error

        information_df['TDTe'][i] = delta_team_test_error
        information_df['TDTr'][i] = delta_team_train_error
        information_df['TDVa'][i] = delta_team_val_error

        information_df['TEPoTe'][i] = post_team_test_error
        information_df['TEPoTr'][i] = post_team_train_error
        information_df['TEPoVa'][i] = post_team_val_error

        information_df['TEPrTe'][i] = pre_team_test_error
        information_df['TEPrTr'][i] = pre_team_train_error
        information_df['TEPrVa'][i] = pre_team_val_error

        information_df['TF'][i] = team_flag

        information_df['TGrDTe'][i] = delta_team_group_test_error
        information_df['TGrDTr'][i] = delta_team_group_train_error
        information_df['TGrDVa'][i] = delta_team_group_val_error

        information_df['TGrEPrTe'][i] = pre_team_group_test_error
        information_df['TGrEPrTr'][i] = pre_team_group_train_error
        information_df['TGrEPrVa'][i] = pre_team_group_val_error

        information_df['TGrEPoTe'][i] = post_team_group_test_error
        information_df['TGrEPoTr'][i] = post_team_group_train_error
        information_df['TGrEPoVa'][i] = post_team_group_val_error

        information_df['TR'][i] = team_pdl.repairs

        information_df['TRDTe'][i] = team_flag*(delta_team_test_error - group_weight_test*delta_team_group_test_error)
        information_df['TRDTr'][i] = team_flag*(delta_team_train_error - group_weight_train*delta_team_group_train_error)
        information_df['TRDVa'][i] = team_flag*(delta_team_val_error - group_weight_val*delta_team_group_val_error)

        information_df['TU'][i] = team_pdl.updates

        
    information_df.to_csv(f"alpha_{alpha}.csv", index = False)

    teams_folder = f"alpha_{alpha}/teams"
    error_tables = defn.generate_errors_table(teams_folder)

if __name__ == "__main__":
    main()