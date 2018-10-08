#import statements
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import os

def h20_init_end(flag):
    if flag == 1:
        h2o.init()
        h2o.remove_all()
    else :
        h2o.cluster.shutdown(prompt=False)



def import_data():
    hw_df = h2o.import_file(os.path.join('Data', 'poisson_sim.csv'))
    hw_df['prog'] = hw_df['prog'].asfactor()
    return hw_df

def create_model(hw_df):
    train, valid, test = hw_df.split_frame([0.7, 0.15], seed=1007)
    hw_df_x = hw_df.col_names[2:]
    hw_df_y = hw_df.col_names[1]
    glm_poisson = H2OGeneralizedLinearEstimator(model_id='glm_v1', family='poisson')
    glm_poisson.train(hw_df_x, hw_df_y, training_frame=train, validation_frame=valid)
    # print(glm_poisson.summary())
    # print('Training Deviance:{0} on {1} Degrees of Freedom '.format(str(glm_poisson.residual_deviance(train=True)), str(glm_poisson.residual_degrees_of_freedom(train=True))))
    # print('Validation Deviance:{0} on {1} Degrees of Freedom '.format(str(glm_poisson.residual_deviance(valid=True)), str(glm_poisson.residual_degrees_of_freedom(valid=True))))
    return (glm_poisson, test)

def predict_model(glm_poisson, test):
    pred = glm_poisson.predict(test)
    pred_scr = pred.round(0)
    # print('Model Performance:', glm_poisson.model_performance(test))
    # glm_poisson.model_performance(test)
    train_score = pred_scr.concat(test)
    print(train_score.head(25))


def main():
    h20_init_end(1)
    hw_df = import_data()
    glm_poisson, test = create_model(hw_df)
    predict_model(glm_poisson, test)
    h20_init_end(0)

if __name__ == '__main__':
    main()