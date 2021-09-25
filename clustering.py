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

data = np.load( 'data_dx.npy', allow_pickle = True )
months = np.load( 'time_dx.npy', allow_pickle = True )
lbl = np.load( 'lbl_dx.npy', allow_pickle = True )

for dis in range( 18 ):
    data = data_dx[dis,:]
    months = time_dx[dis,:]
    lbl = lbl_dx[dis,:]
    column_names = [ 'ID', 'w_mean', 'max_bmi', 'trend', 'num_pos', 'num_neg', 'max_chng', 'median', 'start_bmi_lvl', 'end_bmi_lvl', 'time_to_diagnos', 'age_lvl', 'sex', 'race_eth', 'insurance', 'residence', 'income_cat', 'ldl', 'sys', 'dias', 'a1c' ]
    
    df = pd.DataFrame( columns = column_names )
    
    t0 = time.perf_counter()
    total_samples = 0
    try:
        for visit in range( 2, data.shape[0] ):
            samples = data[visit].shape[0]
            for sample in range( samples ):
                # if total_samples > 1000:
                #     break
                sample_id = int( data[visit][sample, 0, 0] )
                if sample_id == 0:
                    total_samples += 1
                    continue
                # visit_incdnt = lbl[visit][sample,:].nonzero()[0][0]
                # if visit_incdnt == 0:
                #     # print( 'ID: {} diag: {}'.format( sample_id, lbl[visit][sample,:] ))
                #     # print( 'visit_incdnt: {}'.format( visit_incdnt ))
                #     total_samples += 1
                #     continue
                bmi = data[visit][sample,:,1]
                # print( 'ID: {} BMI: {}'.format( sample_id, bmi ))
                if bmi.shape[0] == 1:
                    # print( 'BMI length too short' )
                    total_samples += 1
                    continue
                period = months[visit][sample,:]
                period_diff = np.concatenate( ( [0], np.diff( months[visit][sample,:] ) ) )
                weights = np.reciprocal( period_diff, where = period_diff > 0 ) if np.sum( period_diff ) else np.ones( bmi.shape)
                w_mean = np.average( bmi, weights = weights )
                trend = np.average( np.concatenate( ( [0], np.diff( bmi ) ) ), weights = weights )
                num_pos = np.sum( np.diff( bmi ) > 0 ) / visit
                num_neg = np.sum( np.diff( bmi ) < 0 ) / visit
                max_chng = np.max( np.diff( bmi ))
                median = np.median( bmi )
                max_bmi = np.amax( bmi )
                age_lvl = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_STARTAGE_10Y.iloc[0]
                sex = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_SEX.iloc[0]
                race_eth = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_RACE_ETHNICITY.iloc[0]
                insurance = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_INSURANCE_TYPE.iloc[0]
                residence = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_ZIP_RURALURBAN.iloc[0]
                income_cat = D[D['RANDOM_PATIENT_ID'] == sample_id].PT_ZIP_INCOME_CAT.iloc[0]
                
                ldl = np.mean( D[D['RANDOM_PATIENT_ID'] == sample_id].LABS_LDL_MEAN )
                sys = np.mean( D[D['RANDOM_PATIENT_ID'] == sample_id].VISIT_BP_SYSTOLIC )
                dias = np.mean( D[D['RANDOM_PATIENT_ID'] == sample_id].VISIT_BP_DIASTOLIC )
                a1c = np.mean( D[D['RANDOM_PATIENT_ID'] == sample_id].LABS_A1C_MEAN )
                
                # try:
                #     df.at[sample_id,'w_mean'] = w_mean
                #     df.at[sample_id,'max_bmi'] = max_bmi
                #     df.at[sample_id,'trend'] = trend
                #     df.at[sample_id,'num_pos'] = num_pos
                #     df.at[sample_id,'num_neg'] = num_neg
                #     df.at[sample_id,'max_chng'] = max_chng
                #     df.at[sample_id,'median'] = median
                #     df.at[sample_id,'start_bmi_lvl'] = calculate_bmi_index( bmi[0])
                #     df.at[sample_id,'end_bmi_lvl'] = calculate_bmi_index( bmi[-1])
                #     df.at[sample_id,'time_to_diagnos'] = period[-1]
                # except KeyError:
                # print( 'Creating new row in DataFrame' )
                values = { 'ID' : sample_id, 'w_mean': w_mean, 'max_bmi': max_bmi, 'trend': trend, 'num_pos': num_pos, 'num_neg': num_neg, 'max_chng': max_chng, 'median': median, 'start_bmi_lvl': calculate_bmi_index( bmi[0]), 'end_bmi_lvl': calculate_bmi_index( bmi[-1]), 'time_to_diagnos': period[-1], 'age_lvl': age_lvl, 'sex': sex, 'race_eth': race_eth, 'insurance': insurance, 'residence': residence, 'income_cat': income_cat, 'ldl': ldl, 'sys': sys, 'dias': dias, 'a1c': a1c }
                # print( values )
                # print( 'Diag: {}'.format( lbl[visit][sample,:] ))
                df = df.append( values, ignore_index = True )
                total_samples += 1
                # if not total_samples % 1000:
                #     print( 'Done:', total_samples )
    except ZeroDivisionError:
        print( sample_id, visit )
    # df = df.set_index( 'ID' )
    t1 = time.perf_counter()
    dt = t1 - t0
    print( 'Elapsed time: {:d}h {:d}m {:.3f}s'.format( int( dt / 3600 ), int( ( dt % 3600 ) / 60 ), ( dt % 3600 ) % 60 ) )
    filename = dx_col[dis] + '_dynsta.csv'
    df.to_csv( filename, index = False )
    print( filename, 'saved', df.shape )
    
# ====================================================== clustering ===================================================
k = 5
model = KMeans()
visualizer = KElbowVisualizer( model, metric = 'calinski_harabasz', locate_elbow = True, k=(2,10) )
visualizer.fit( df[df.columns[1:]] )        # Fit the data to the visualizer
k = visualizer.elbow_value_
kmeans = KMeans(n_clusters = k, random_state=0, init='k-means++').fit( df[df.columns[1:]] )
centroid_values = kmeans.cluster_centers_
c_labels = kmeans.labels_
unique_labels, counts = np.unique( c_labels, return_counts = True )
print( counts )
df['cluster_lbl'] = pd.Series( c_labels )

plt.figure(figsize=(12,12))
plt.plot( np.mean( df[df['cluster_lbl']==0].start_bmi_lvl), np.mean( df[df['cluster_lbl']==0].end_bmi_lvl), 'rX', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==1].start_bmi_lvl), np.mean( df[df['cluster_lbl']==1].end_bmi_lvl), 'bX', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==2].start_bmi_lvl), np.mean( df[df['cluster_lbl']==2].end_bmi_lvl), 'gX', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==3].start_bmi_lvl), np.mean( df[df['cluster_lbl']==3].end_bmi_lvl), 'kX',  markersize = 20 )
plt.xlabel('start_bmi_level')
plt.ylabel('end_bmi_level')

plt.figure(figsize=(12,12))
plt.plot( np.mean( df[df['cluster_lbl']==0].time_to_diagnos), np.mean( df[df['cluster_lbl']==0].w_mean), 'r*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==1].time_to_diagnos), np.mean( df[df['cluster_lbl']==1].w_mean), 'b*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==2].time_to_diagnos), np.mean( df[df['cluster_lbl']==2].w_mean), 'g*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==3].time_to_diagnos), np.mean( df[df['cluster_lbl']==3].w_mean), 'k*',  markersize = 20 )
plt.xlabel('time_to_diagnos')
plt.ylabel('w_mean')

plt.figure(figsize=(12,12))
plt.plot( np.mean( df[df['cluster_lbl']==0].time_to_diagnos), np.mean( df[df['cluster_lbl']==0].max_bmi), 'r*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==1].time_to_diagnos), np.mean( df[df['cluster_lbl']==1].max_bmi), 'b*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==2].time_to_diagnos), np.mean( df[df['cluster_lbl']==2].max_bmi), 'g*', markersize = 20 )
plt.plot( np.mean( df[df['cluster_lbl']==3].time_to_diagnos), np.mean( df[df['cluster_lbl']==3].max_bmi), 'k*',  markersize = 20 )
plt.xlabel('time_to_diagnos')
plt.ylabel('max_bmi')