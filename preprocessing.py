# -*- coding: utf-8 -*-
"""
@author: Dip
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency, norm

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
#from sklearn.externals.joblib import dump, load
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
#from yellowbrick.cluster import KElbowVisualizer 
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def calculate_bmi_index( bmi ):
    if bmi < 18.5:
        return 0
    elif 18.5 <= bmi < 25:
        return 1
    elif 25 <= bmi < 30:
        return 2
    else:
        return 3
    
def get_age_lvl( age ):
    if age == '<30':
        return 0
    elif age == '30-39':
        return 1
    elif age == '40-49':
        return 2
    elif age == '50-59':
        return 3
    elif age == '60-69':
        return 4
    elif age == '70+':
        return 5
    
def read_data():
    cols = [ 'RANDOM_PATIENT_ID', 
        'VISIT_YEAR', 
        'VISIT_MONTH', 
        'VISIT_BMI',
        'PT_STARTAGE_10Y',
        'PT_SEX',
        'PT_RACE_ETHNICITY',
        'PT_INSURANCE_TYPE',
        'PT_ZIP_RURALURBAN',
        'PT_ZIP_INCOME_CAT',
        'LABS_LDL_MEAN',
        'VISIT_BP_SYSTOLIC',
        'VISIT_BP_DIASTOLIC',
        'LABS_A1C_MEAN',
        'DX_HYPOTHYROIDISM',
        'DX_STROKE',
        'DX_ALZ_DEM',
        'DX_ANEMIA',
        'DX_ASTHMA',
        'DX_AFIB',
        'DX_CARDIAC',
        'DX_BPH',
        'DX_CKD',
        'DX_COPD',
        'DX_CANCER',
        'DX_DEPRESSION',
        'DX_DIABETES',
        'DX_HIPFX_OSTEO',
        'DX_HYPERLIPID',
        'DX_HYPERTENS',
        'DX_OBESITY',
        'DX_ARTHRITIS', 
        'PRIORDX_HYPOTHYROIDISM',
        'PRIORDX_STROKE',
        'PRIORDX_ALZ_DEM',
        'PRIORDX_ANEMIA',
        'PRIORDX_ASTHMA',
        'PRIORDX_AFIB',
        'PRIORDX_CARDIAC',
        'PRIORDX_BPH',
        'PRIORDX_CKD',
        'PRIORDX_COPD',
        'PRIORDX_CANCER',
        'PRIORDX_DEPRESSION',
        'PRIORDX_DIABETES',
        'PRIORDX_HIPFX_OSTEO',
        'PRIORDX_HYPERLIPID',
        'PRIORDX_HYPERTENS',
        'PRIORDX_OBESITY',
        'PRIORDX_ARTHRITIS', 
        ]
    dx = pd.DataFrame()
    for i in range( 1, 13 ):
        filename = "hd4a_data_ptview_final_" + str( i ) + ".csv"
        df = pd.read_csv( filename )
        dy = df[ cols ].copy()
        dy = dy.dropna( axis = 0, how = 'any' )
        del df
        dx = pd.concat( ( dx, dy ) )
        print( "Reading file", filename, ": Success" )
    return dx

dx_col = [ 'DX_HYPOTHYROIDISM', 'DX_STROKE', 'DX_ALZ_DEM', 'DX_ANEMIA', 'DX_ASTHMA', 'DX_AFIB', 'DX_CARDIAC', 'DX_BPH', 'DX_CKD', 'DX_COPD', 'DX_CANCER', 'DX_DEPRESSION', 'DX_DIABETES', 'DX_HIPFX_OSTEO', 'DX_HYPERLIPID', 'DX_HYPERTENS', 'DX_OBESITY', 'DX_ARTHRITIS' ]

t0 = time.perf_counter()
dx = read_data()
t1 = time.perf_counter()
dt = t1 - t0
print( 'Elapsed time: {:d}h {:d}m {:.3f}s'.format( int( dt / 3600 ), int( ( dt % 3600 ) / 60 ), ( dt % 3600 ) % 60 ) )

print( "Feature extraction DONE" )
bmi = dx['VISIT_BMI'].replace([ 'high', 'low' ], [ 49.1, 19.0 ] )
dx = dx.assign( VISIT_BMI = bmi.astype( float ) )
a1c = dx['LABS_A1C_MEAN'].replace( [ 'high', 'low' ], [ 11.7, 5 ] )
dx = dx.assign( LABS_A1C_MEAN = a1c.astype( float ) )
ldl = dx['LABS_LDL_MEAN'].replace( [ 'high', 'low' ], [ 149, 44 ] )
dx = dx.assign( LABS_LDL_MEAN = ldl.astype( float ) )
sys = dx['VISIT_BP_SYSTOLIC'].replace( [ 'high', 'low' ], [ 166, 98 ] )
dx = dx.assign( VISIT_BP_SYSTOLIC = sys.astype( float ) )
dias = dx['VISIT_BP_DIASTOLIC'].replace( [ 'high', 'low' ], [ 100, 58 ] )
dx = dx.assign( VISIT_BP_DIASTOLIC = dias.astype( float ) )
dx = dx.dropna( axis = 0, how = 'any' )
dx = dx.sort_values( by = ['RANDOM_PATIENT_ID','VISIT_YEAR','VISIT_MONTH']).reset_index( drop = True )
print( "Dropping NA DONE" )

D = dx.copy()

samples = D.RANDOM_PATIENT_ID.nunique()
p = np.linspace( int( samples * 0.1 ), samples, 10, dtype = int )
p_ind = 0    
print( "Total samples:", samples )

t0 = time.perf_counter()
time_dx = np.zeros( ( 18, 99 ), dtype = object )
data_dx = np.zeros( ( 18, 99 ), dtype = object )
lbl_dx = np.zeros( ( 18, 99 ), dtype = object )

n_features = D.shape[1]
for i in range( 99 ):
    for j in range( 18 ):
        data_dx[j,i] = np.zeros( ( 1, i, 3 ) )  
        time_dx[j,i] = np.zeros( ( 1, i ))
        lbl_dx[j,i] = np.zeros( ( 1, i ))
    
index = 0
count = 0
visits = np.empty( ( 18, ), dtype = object )
dis_total = []
for n in range( 18 ):
    visits[n] = []
#     s = 0
#     for v in range( 99 ):
#         s += data_dx[n,v].shape[0]
#     dis_total.append( s )
for n, d in D.groupby( [ 'RANDOM_PATIENT_ID' ] ):
    n_visit = d.shape[0]
    tm = []
    i = -1
    for idx in range( n_visit ):
        if idx == 0:
            t = 0
        else:    
            t = d['VISIT_MONTH'].iloc[idx] - d['VISIT_MONTH'].iloc[idx - 1]
            y = d['VISIT_YEAR'].iloc[idx] - d['VISIT_YEAR'].iloc[idx - 1]
            if y > 0:
                t = y * 12 + t
        tm.append( t )
    d.loc[:,'TIME_DIF'] = tm
 
    chg = []
    for j in range( 14, 32 ):
        chg.append( max( d.iloc[:,j] ) - int( d.iloc[idx,j + 18] ) )
    # print( chg )
    
    #     if f_inc >= 0 and sum( d.iloc[:,j] ) >= 0.75 * n_visit and not d.iloc[0,j+18]:
    for j in range( 14, 32 ):
        i = j - 14
        # if not chg[i] > 0 and not max( d.iloc[:,j] ):                                                       # modified for negative patients
        if chg[i] > 0:
            diag = d.iloc[:,j]
            # print( diag )
            if n_visit in visits[i]:
                # if data_dx[i,n_visit].shape[0] > dis_total[i]:
                #     continue
                data_dx[i,n_visit] = np.append( data_dx[i,n_visit], np.zeros(( 1, n_visit, 3 )), axis = 0 )
                time_dx[i,n_visit] = np.append( time_dx[i,n_visit], np.zeros(( 1, n_visit )), axis = 0 )
                lbl_dx[i,n_visit] = np.append( lbl_dx[i,n_visit], np.zeros(( 1, n_visit )), axis = 0 )
                sz = data_dx[i,n_visit].shape[0]
                data_dx[i,n_visit][sz-1] = d[['RANDOM_PATIENT_ID', 'TIME_DIF', 'VISIT_BMI' ]]
                time_dx[i,n_visit][sz-1] = np.array( tm ).reshape( 1, n_visit )
                lbl_dx[i,n_visit][sz-1] = np.array( diag ).reshape(( 1, n_visit ))
                # print( "append" )
            else:
                data_dx[i,n_visit][0] = d[['RANDOM_PATIENT_ID', 'TIME_DIF', 'VISIT_BMI' ]]
                time_dx[i,n_visit][0] = np.array( tm ).reshape( 1, n_visit )
                lbl_dx[i,n_visit][0] = np.array( diag ).reshape(( 1, n_visit ))
                visits[i].append( n_visit )
                # print( "first_entry" )
    count += 1
    if count == p[p_ind]:
        print( "Completed:", str( ( p_ind + 1 ) * 10 ) + "%", "samples:", count )
        p_ind += 1
t1 = time.perf_counter()
dt = t1 - t0
print( 'Elapsed time: {:d}h {:d}m {:.3f}s'.format( int( dt / 3600 ), int( ( dt % 3600 ) / 60 ), ( dt % 3600 ) % 60 ) )

np.save( 'data_dx.npy', data_dx, allow_pickle = True )
np.save( 'time_dx.npy', time_dx, allow_pickle = True )
np.save( 'lbl_dx.npy', lbl_dx, allow_pickle = True )