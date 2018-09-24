import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
import numpy as np
import os
h2o.init()
h2o.remove_all()

hw_df = h2o.import_file(os.path.realpath('.\\Data\poisson_sim.csv'))
# print(hw_df.head(10))
hw_df['prog'] = hw_df['prog'].asfactor()
train, valid, test = hw_df.split_frame([0.7, 0.15], seed=1007)
hw_df_x = hw_df.col_names[2:]
hw_df_y = hw_df.col_names[1]

glm_poisson = H2OGeneralizedLinearEstimator(model_id='glm_v1', family='poisson')
glm_poisson.train(hw_df_x, hw_df_y, training_frame=train, validation_frame=valid)
glm_poisson
