#import statements
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
import numpy as np
import os

def h20_init_end(flag):
    if flag == 1:
        h2o.init()
        h2o.remove_all()
    else :
        h2o.cluster.shutdown(prompt=False)



def import_data():
    hw_df = h2o.import_file(os.path.realpath('.\\Data\poisson_sim.csv'))
    hw_df['prog'] = hw_df['prog'].asfactor()
    return hw_df

def create_model(hw_df):
    train, valid, test = hw_df.split_frame([0.7, 0.15], seed=1007)
    hw_df_x = hw_df.col_names[2:]
    hw_df_y = hw_df.col_names[1]
    glm_poisson = H2OGeneralizedLinearEstimator(model_id='glm_v1', family='poisson')
    glm_poisson.train(hw_df_x, hw_df_y, training_frame=train, validation_frame=valid)

def main():
    h20_init_end(1)
    hw_df = import_data()
    create_model(hw_df)
    # h20_init_end(0)

if __name__ == '__main__':
    main()