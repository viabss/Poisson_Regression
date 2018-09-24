import pandas as pd
import statsmodels.api as sm
import argparse
import numpy as np
import matplotlib.pyplot as mp
import patsy as pat

def import_dat(location):
    df = pd.read_csv(location, index_col='id')
    df['prog'] = df['prog'].astype('category')
    df['prog'] = df['prog'].cat.rename_categories({1: 'General', 2: 'Academic', 3: 'Vocational'})
    y, x = pat.dmatrices('num_awards~ math + prog', data=df, return_type='dataframe')
    # print(pd.merge(y[:20], x[:20], how='inner', on='id'))
    pois_model = sm.GLM(y, x, family=sm.families.Poisson(sm.families.links.log))
    pois_res = pois_model.fit()
    print(pois_res.summary())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location', help='Location of the File')
    args = parser.parse_args()
    import_dat(args.location)

if __name__ == '__main__':
    main()
