import numpy as np, pandas as pd
import os,glob, re

dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
    pd.read_csv(fn)for fn in glob.glob('../input/*.csv')}
print('data frames read:{}'.format(list(dfs.keys())))

print('local variables with the same names are created.')
for k, v in dfs.items(): locals()[k] = v


print("Raw shape of each dataset")
for k, v in dfs.items(): print("%s : "%k,v.shape)

print("Split id column in sample_submission")
sample_submission["air_store_id"],sample_submission["visit_date"] = sample_submission.id.str[:20],sample_submission.id.str[21:]
sample_submission.head()

print("Unique store Ids in each dataset")
for k, v in dfs.items(): 
    try:       
        print(k," - Unqiue air_stores: ",v.air_store_id.nunique())
    except:
        pass
    try:
        print(k," - Unqiue hpg_stores: ",v.hpg_store_id.nunique())
    except:
        pass

air_reserve['visit_date'] = pd.to_datetime(air_reserve['visit_datetime']).dt.date.astype(str)

reserve_summary = air_reserve.groupby(['air_store_id','visit_date'])['reserve_visitors'].sum().reset_index()

new_train = air_visit_data.merge(reserve_summary, on =['air_store_id','visit_date'],how = 'left').fillna(0)

new_train['walkins'] = new_train['visitors'] - new_train['reserve_visitors']
new_train.loc[new_train['walkins'] <0,'walkins'] = 0
new_train['noshows'] = new_train['reserve_visitors'] - new_train['visitors']
new_train.loc[new_train['noshows'] <0,'noshows'] = 0
new_train.head()

weekdayholidays = date_info.apply(lambda x: x.day_of_week in ['Saturday','Sunday'] and x.holiday_flg == 1,axis=1)
date_info.loc[weekdayholidays,'holiday_flg'] = 0

date_info['weights'] = ((date_info.index + 1)/ len(date_info))**7

new_train = new_train.merge(date_info,left_on = 'visit_date',right_on = 'calendar_date', how ='left').drop('calendar_date',axis = 1)

new_train['visitors'] = new_train['visitors'].apply(pd.np.log1p)
new_train['reserve_visitors'] = new_train['reserve_visitors'].apply(pd.np.log1p)
new_train['walkins'] = new_train['walkins'].apply(pd.np.log1p)
new_train['noshows'] = new_train['noshows'].apply(pd.np.log1p)

weighted_mean_visitors = lambda x : ((x.visitors * x.weights).sum() / (x.weights).sum())
visitors_per_weekday = new_train.groupby(['air_store_id','day_of_week','holiday_flg']).apply(weighted_mean_visitors).reset_index()

weighted_mean_reservations = lambda x : ((x.reserve_visitors * x.weights).sum() / (x.weights).sum())
reserves_per_weekday = new_train.groupby(['air_store_id','day_of_week','holiday_flg']).apply(weighted_mean_reservations).reset_index()
reserves_per_weekday.head()

weighted_mean_walkins = lambda x : ((x.walkins * x.weights).sum() / (x.weights).sum())
walkin_visitors_per_weekday = new_train.groupby(['air_store_id','day_of_week','holiday_flg']).apply(weighted_mean_walkins).reset_index()

weighted_mean_noshows = lambda x : ((x.noshows * x.weights).sum() / (x.weights).sum())
noshows_per_weekday = new_train.groupby(['air_store_id','day_of_week','holiday_flg']).apply(weighted_mean_noshows).reset_index()

summarized_train = visitors_per_weekday.merge(
    reserves_per_weekday, on= ['air_store_id','day_of_week','holiday_flg'],how = 'outer')

summarized_train.rename(columns={'0_x':'wt_visitors','0_y':'wt_reserves'},inplace = True)

summarized_train = summarized_train.merge(
    walkin_visitors_per_weekday, on = ['air_store_id','day_of_week','holiday_flg'],how ='outer')

summarized_train = summarized_train.merge(
    noshows_per_weekday, on= ['air_store_id','day_of_week','holiday_flg'],how = 'outer')

summarized_train.rename(columns={'0_x':'walkins','0_y':'noshows'},inplace = True)


test = sample_submission.merge(date_info,left_on='visit_date',right_on='calendar_date',how = 'left').drop(['calendar_date','weights'],axis = 1)

newtest = test.merge(reserve_summary, on=['air_store_id','visit_date'], how='left').fillna(0)

newtest = newtest.merge(summarized_train,on = ['air_store_id','day_of_week','holiday_flg'], how = 'left')

temp = newtest[newtest.wt_visitors.isnull()].merge(summarized_train[summarized_train.holiday_flg == 0]
                                            ,on = ['air_store_id','day_of_week'], how = 'left')


newtest.loc[newtest.wt_visitors.isnull(),'wt_visitors'] = temp['wt_visitors_y'].values
newtest.loc[newtest.wt_reserves.isnull(),'wt_reserves'] = temp['wt_reserves_y'].values
newtest.loc[newtest.walkins.isnull(),'walkins'] = temp['walkins_y'].values
newtest.loc[newtest.noshows.isnull(),'noshows'] = temp['noshows_y'].values


temp2 = newtest[newtest.wt_visitors.isnull()].merge(summarized_train[[
    'air_store_id','wt_visitors','wt_reserves','walkins','noshows']].groupby('air_store_id').mean().reset_index(),
                                                    on = 'air_store_id',how = "left")

newtest.loc[newtest.wt_visitors.isnull(),'wt_visitors'] = temp2['wt_visitors_y'].values
newtest.loc[newtest.wt_reserves.isnull(),'wt_reserves'] = temp2['wt_reserves_y'].values
newtest.loc[newtest.walkins.isnull(),'walkins'] = temp2['walkins_y'].values
newtest.loc[newtest.noshows.isnull(),'noshows'] = temp2['noshows_y'].values


max_visitors = air_visit_data.groupby('air_store_id')['visitors'].max().reset_index()
max_visitors.rename(columns = {'visitors' : 'max_cap'},inplace = True)

newtest = newtest.merge(max_visitors,on= 'air_store_id', how = 'left')

newtest.drop(['visitors','air_store_id','visit_date','day_of_week','holiday_flg'],inplace = True,axis = 1)

newtest['wt_visitors'] = newtest['wt_visitors'].apply(pd.np.expm1)
newtest['wt_reserves'] = newtest['wt_reserves'].apply(pd.np.expm1)
newtest['walkins'] = newtest['walkins'].apply(pd.np.expm1)
newtest['noshows'] = newtest['noshows'].apply(pd.np.expm1)

newtest['calculated_visits'] = ((newtest['reserve_visitors']+newtest['wt_reserves'])/2) +newtest['walkins'] - newtest['noshows']

#newtest['visitors'] = newtest['calculated_visits']

k = .4

newtest['visitors'] = ((newtest['wt_visitors'] * k) + ((1-k)*newtest['calculated_visits']))


# newtest.loc[newtest['visitors'] > newtest['max_cap'],'visitors'] = newtest['max_cap']


newtest.loc[newtest['visitors'] < 0,'visitors'] = newtest['wt_reserves']


result = newtest[['id','visitors']]


result.to_csv('result_dump444.csv', float_format='%.4f', index=None)

# newsub = sub_merge[['id','visitors_x','visitors_y']].merge(result,how = 'inner',on = 'id')


# newsub['visitors'] = (newsub['visitors_x'] + newsub['visitors_y']*1.1 + newsub['visitors'])/3

# newsub[['id', 'visitors']].to_csv('submission66.csv', index=False)