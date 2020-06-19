import unittest
import geone
import pandas as pd

class TestDeesseClassifier(unittest.TestCase):
    def setUp(self):
        DATA_DIR = 'data/'
        ti = geone.img.readImageGslib(DATA_DIR+'A.gslib')
        self.data = pd.DataFrame(geone.img.readPointSetGslib(DATA_DIR+'sample_100.gslib').to_dict())
        self.deesse_estimator = geone.deesseinterface.DeesseClassifier(
            varnames = ['X','Y','Z', 'facies'],
            nx=100, ny=100, nz=1,     # dimension of the simulation grid (number of cells)
            sx=1.0, sy=1.0, sz=1.0,   # cells units in the simulation grid (here are the default values)
            ox=0.0, oy=0.0, oz=0.0,   # origin of the simulation grid (here are the default values)
            nv=1, varname='facies',   # number of variable(s), name of the variable(s)
            nTI=1, TI=ti,           # number of TI(s), TI (class dsi.Img)
            distanceType=0,           # distance type: proportion of mismatching nodes (categorical var., default)
            nneighboringNode=20,      # max. number of neighbors (for the patterns)
            distanceThreshold=0.1,    # acceptation threshold (for distance between patterns)
            maxScanFraction=0.25,     # max. scanned fraction of the TI (for simulation of each cell)
            npostProcessingPathMax=1, # number of post-processing path(s)
            seed=20191201,            # seed (initialization of the random number generator)
            nrealization=2,           # number of realization(s)
            nthreads=2)               # number of threads to use for simulations

    def test_sklearn_compatible(self):
        X, y = self.data[['X', 'Y', 'Z']], self.data['code_real00000']
        clf = self.deesse_estimator
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'X_')
        assert hasattr(clf, 'y_')

        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],)

class TestDeesseRegressor(unittest.TestCase):
    def setUp(self):
        DATA_DIR = 'data/'
        ti = geone.img.readImageGslib(DATA_DIR+'tiContinuous.gslib')
        self.data = pd.DataFrame(geone.img.readPointSetGslib(DATA_DIR+'continous_sample_100.gslib').to_dict())
        self.deesse_regressor = geone.deesseinterface.DeesseRegressor(
            varnames = ['X','Y','Z', 'facies'],
            nx=100, ny=100, nz=1,     # dimension of the simulation grid (number of cells)
            sx=1.0, sy=1.0, sz=1.0,   # cells units in the simulation grid (here are the default values)
            ox=0.0, oy=0.0, oz=0.0,   # origin of the simulation grid (here are the default values)
            nv=1, varname='facies',   # number of variable(s), name of the variable(s)
            nTI=1, TI=ti,           # number of TI(s), TI (class dsi.Img)
            distanceType=1,           # distance type: proportion of mismatching nodes (categorical var., default)
            nneighboringNode=20,      # max. number of neighbors (for the patterns)
            distanceThreshold=0.1,    # acceptation threshold (for distance between patterns)
            maxScanFraction=0.25,     # max. scanned fraction of the TI (for simulation of each cell)
            npostProcessingPathMax=1, # number of post-processing path(s)
            seed=20191201,            # seed (initialization of the random number generator)
            nrealization=2,           # number of realization(s)
            nthreads=2)               # number of threads to use for simulations

    def test_sklearn_compatible(self):
        X, y = self.data[['X', 'Y', 'Z']], self.data['facies_real00000']
        clf = self.deesse_regressor
        clf.fit(X, y)
        assert hasattr(clf, 'is_fitted_')
        assert hasattr(clf, 'X_')
        assert hasattr(clf, 'y_')

        y_pred = clf.sample_y(X)
        assert y_pred.shape == (X.shape[0], 2)
