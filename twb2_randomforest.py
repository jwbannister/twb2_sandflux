import os
import pandas as pd
import numpy as np
import dfgui
import math
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import cross_val_score
from sklearn import metrics
import seaborn as sns
import jwb_data
import jwb_general

surface_survey_query = "SELECT *, " +\
        "SUBSTRING(site, 1, 4) AS site_num " +\
        "FROM field_data.twb2_qa_survey"
surface_survey_df = pull_owens_data(surface_survey_query).\
        drop(['rs', 'fd', 'is', 'rh'], axis=1)
ss_complete = surface_survey_df[surface_survey_df.isnull().sum(axis=1)==0]
ss_df = ss_complete.groupby(['area', 'site_num', 'yr_mo']).mean()
ss_df.reset_index(inplace=True)

yrmo_string = "('" + "', '".join(set(ss_df.yr_mo)) + "')"
csc_string = "('" + "', '".join(set([item[0:4] for item in ss_df.site_num])) + "')"
flux_query = "SELECT datetime, csc, ws_10m, wd_10m, sand_flux, " +\
        "SUBSTRING(CONCAT(EXTRACT(year from datetime::timestamp), '_', " +\
        "EXTRACT(month from datetime::timestamp)), 3, 5) AS yr_mo " +\
        "FROM sensit.validated_5min_sandflux " +\
        "WHERE SUBSTRING(CONCAT(EXTRACT(year from datetime::timestamp), '_', " +\
        "EXTRACT(month from datetime::timestamp)), 3, 5) IN " + yrmo_string + " " +\
        "AND csc IN " + csc_string + " " +\
        "AND NOT invalid " +\
        "AND dwp_mass>=0;"
flux_df = pull_owens_data(flux_query)
df1 = pd.merge(flux_df, ss_df, how='left', left_on=['csc', 'yr_mo'], \
        right_on=['site_num', 'yr_mo'])
df1['date'] = (df1.datetime - datetime.timedelta(seconds=1)).dt.date
df1['hour'] = (df1.datetime - datetime.timedelta(seconds=1)).dt.hour
df2 = df1[df1.isnull().sum(axis=1)==0]


    shifted = [shift_dict[get_q(a)] for a in lst]
    quadrent = np.digitize(lst, [0, 90, 180, 270, 360.1])
    i = [math.cos(a) for a in lst]

group_functions = {'ws_10m': 'mean', 'wd_10m': 'median', 

def test_add(x):
    y = x + 1
    return y


df2.loc[:, 'is_training'] = np.random.uniform(0, 1, len(df2)) <= 0.75
train, test = df2[df2['is_training']==True], df2[df2['is_training']==False]
ss_features = ['rs_avg', 'rh_avg', 'rs_rh', 'clods']

ss_clf = RandomForestClassifier()
y, index = ss_train.area.cat.codes, ss_train.area.cat.categories
ss_clf.fit(ss_train[ss_features], y)

preds = pd.Categorical(index[ss_clf.predict(ss_test[ss_features])], index)
pd.crosstab(ss_test['area'], preds, rownames=['actual'], colnames=['preds'], \
        dropna=False)
zip(ss_train[ss_features], ss_clf.feature_importances_)
metrics.accuracy_score(ss_test['area'], preds)



df2 = df1[df1.isnull().sum(axis=1)==0].drop_duplicates()
df2.groupby(['csc', 'date']).sand_flux.sum()

df3 = df2[df2.sand_flux>0]

area_categories = pd.Categorical(df3.area).categories
df3['is_training'] = np.random.uniform(0, 1, len(df3)) <= 0.75
train, test = df3[df3['is_training']==True], df3[df3['is_training']==False]
features = ['rs_avg', 'rh_avg', 'rs_rh', 'clods', 'ws_10m', 'wd_10m', 'area']

clf = RandomForestRegressor()
y = train.sand_flux
clf.fit(train[features], y)
preds = clf.predict(test[features])
zip(train[features], clf.feature_importances_)
metrics.r2_score(test['sand_flux'], preds)

axes = plt.gca()
axes.plot([0, 1])
axes.scatter(test['sand_flux'], preds)
axes.set_xlim([0, 0.1])
axes.set_ylim([0, 0.1])
