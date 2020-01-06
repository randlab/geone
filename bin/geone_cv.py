#! /usr/bin/env python3


import argparse
import json
import os
import sys

import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from geone.cv_metrics import brier_score, zero_one_score, balanced_linear_score, SkillScore
from geone.deesseinterface import DeesseEstimator
from geone.img import readImageGslib
from geone.gslib import read

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file')

    args = parser.parse_args()
    run_cv(args.input)

def run_cv(input_filename):
    with open(input_filename, 'r') as input_file:
        parameters = json.load(input_file)

    observations = parameters['observations']

    observations_df = pd.DataFrame(read(observations['file']))

    cross_validator = CrossValidator(
            results_path=parameters['results_path'],
            estimator_parameters=parameters['estimator'],
            scoring = parameters['scoring'],
            cv_splitter_parameters=parameters['cv_splitter'],
            model_selector_parameters=parameters['model_selector'],
            )

    X =observations_df[observations['coordinates']]
    y = observations_df[observations['variable']]
    cross_validator.fit(X=X, y=y)
    cross_validator.write_pandas_results()


class CrossValidator():
    def __init__(self,
            results_path,
            estimator_parameters,
            scoring,
            cv_splitter_parameters,
            model_selector_parameters,
            ):

        # Preprocess scoring dictionary (recognize geone functions)
        for key, scoring_method in scoring.items():
            # potentially dangerous implementation but it's not a critical script
            try:
                scoring[key] = eval(scoring_method)
            except NameError:
                # expect string here, sklearn's scoring method
                pass
        self.scoring = scoring

        # Convert TI filename to Image
        for key, parameter in estimator_parameters.items():
            if key == 'TI':
                estimator_parameters[key] = readImageGslib(parameter)

        # In model_selector param_grid
        # Convert all TI filenames to Image
        for key, parameter in model_selector_parameters["param_grid"].items():
            if key == 'TI':
                # it's a list
                model_selector_parameters["param_grid"]["TI"] = [
                        readImageGslib(x) for x in parameter]


        # Set estimator and scikit learn's engine
        self.estimator = DeesseEstimator(**estimator_parameters)
        self.cv_splitter = StratifiedKFold(**cv_splitter_parameters)
        self.model_selector = GridSearchCV(estimator=self.estimator,
                scoring=self.scoring,
                cv=self.cv_splitter,
                **model_selector_parameters)

        # Prepare the results directory
        try:
            os.makedirs(results_path)
        except FileExistsError:
            sys.exit("Directory {} exists. "
                    "Remove it or specify another results_path in your input".format(results_path))
        self.results_path = results_path + '/cv_results.csv'

    def fit(self, X, y):
        self.model_selector.fit(X,y)

    def write_pandas_results(self):
        pd.DataFrame(self.model_selector.cv_results_).to_csv(self.results_path)
        
if __name__=='__main__':
    main()
