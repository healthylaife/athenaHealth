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

static_var = [ 'age_lvl', 'sex', 'race_eth', 'insurance', 'residence', 'income_cat', 'ldl', 'sys', 'dias', 'a1c', 'true_label' ]
dx_col = [ 'DX_HYPOTHYROIDISM', 'DX_STROKE', 'DX_ALZ_DEM', 'DX_ANEMIA', 'DX_ASTHMA', 'DX_AFIB', 'DX_CARDIAC', 'DX_BPH', 'DX_CKD', 'DX_COPD', 'DX_CANCER', 'DX_DEPRESSION', 'DX_DIABETES', 'DX_HIPFX_OSTEO', 'DX_HYPERLIPID', 'DX_HYPERTENS', 'DX_OBESITY', 'DX_ARTHRITIS' ]
enc = OrdinalEncoder()

def survey( results, category_names, counts ):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list( results.keys() )
    data = np.array( list( results.values() ))
    data_cum = data.cumsum( axis = 1 )
    category_colors = plt.get_cmap( 'RdYlGn_r' )(
        np.linspace( 0.15, 0.85, data.shape[1] ))

    fig, ax = plt.subplots( figsize = ( 10, 3 ))
    ax.invert_yaxis()
    ax.xaxis.set_visible( False )
    ax.set_xlim( 0, np.sum(data, axis = 1 ).max() )
    # ax.xticks( [ 0, 1 ] )

    for i, ( colname, color ) in enumerate( zip( category_names, category_colors )):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left = starts, height = 0.75,
                label = colname, color = color)
        xcenters = starts + widths / 2
        
        font = {'family': 'sans serif',
            'color':  'darkred',
            'weight': 'bold',
            'size': 16,
            }
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, ( x, c ) in enumerate( zip ( xcenters, widths )):
            ax.text( x, y, str( int( c * counts[y] ) ) + ' ± ' + str( con[y,i] ), ha = 'center', va = 'center',
                    color = text_color, fontsize = 'large', fontdict = font )
    ax.legend( ncol = len( category_names ), bbox_to_anchor = ( 0.55, 1 ),
              loc = 'lower left', fontsize = 'large' )
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    textstr = 'Silhouette Score: ' + str( s_score ) + '    Calinski-Harabasz Score: ' + str( c_score )
    ax.text(0, -0.05, textstr, transform = ax.transAxes, fontsize = 14,
            verticalalignment = 'top', bbox = props)

    return fig, ax

for j in range( 7 ):
    for col in dx_col:
        posfilename = col + '_dynsta.csv'
        dx = pd.read_csv( posfilename )
        dx = dx.drop( columns = [ 'time_to_diagnos' ] )
        dx = dx.set_index( 'ID' )
        dx['true_label'] = 1
        dx = dx.round( 2 )
        #negfilename = col + '_' + str(j) + '_neg.csv'
        negfilename = col + '_neg.csv'
        dy = pd.read_csv( negfilename )
        # dy = dy.drop( columns = [ 'time_to_diagnos' ] )
        dy = dy.set_index( 'ID' )
        dy['true_label'] = 0
        dy = dy.round( 2 )
        d = pd.concat( ( dx, dy ), axis = 0 )
        d.index = d.index.astype( int )
        d = d.sample(frac=1).reset_index(drop=False)               # shuffling the samples
        # d = d.set_index( 'index' )
        # d = d.rename_axis( 'ID' )
        lbl_n = len( np.unique( d['true_label']))
        samples = d.shape[0]
        d = d.dropna()
        if d.shape[0] < samples:
            print( 'Samples reduced:', samples - d.shape[0] )
        #====================================== For table description in paper ===========================================
        print( col )
        u, c = np.unique( d.age_lvl, return_counts = True )
        print( "Dominanat age group:", u[argmax( c )])
        u, c = np.unique( d.sex, return_counts = True )
        print( "% Male:", round( c[1] / sum(c) * 100, 2))
        u, c = np.unique( d.race_eth, return_counts = True )
        print( "% White:", round( c[3] / sum(c) * 100, 2 ))
        print( "----------------------------------------" )
        #=================================================================================================================
        # x_ = TSNE( n_components = 2 ).fit_transform( d[d.columns[1:10]] )
        
        models = {
            'kmeans': KMeans(),
            'agg_ward': AgglomerativeClustering(),
            'agg_cmplt': AgglomerativeClustering(),
            'agg_avg': AgglomerativeClustering(),
            'agg_sngl': AgglomerativeClustering()
            }
        
        for model_key, model_value in models.items():
            model = model_value            
            visualizer = KElbowVisualizer( model, metric = 'calinski_harabasz', locate_elbow = True, k=(2,10) )
            try:
                visualizer.fit( d[d.columns[1:10]] )        # Fit the data to the visualizer
            except MemoryError:
                continue
            k = visualizer.elbow_value_ if visualizer.elbow_value_ is not None else 2
            # k = 2
            if model_key == 'kmeans':
                kmeans = KMeans( n_clusters = k, random_state = 0, n_jobs = -1 ).fit( d[d.columns[1:10]] )
                c_labels = kmeans.labels_
            elif model_key == 'agg_ward':
                agg = AgglomerativeClustering( n_clusters = k, linkage = 'ward' ).fit( d[d.columns[1:10]] )
                c_labels = agg.labels_
            elif model_key == 'agg_cmplt':
                agg = AgglomerativeClustering( n_clusters = k, linkage = 'complete' ).fit( d[d.columns[1:10]] )
                c_labels = agg.labels_
            elif model_key == 'agg_avg':
                agg = AgglomerativeClustering( n_clusters = k, linkage = 'average' ).fit( d[d.columns[1:10]] )
                c_labels = agg.labels_
            elif model_key == 'agg_sngl':
                agg = AgglomerativeClustering( n_clusters = k, linkage = 'single' ).fit( d[d.columns[1:10]] )
                c_labels = agg.labels_
            unique_labels, counts = np.unique( c_labels, return_counts = True )
            print( counts )
            d['cluster_label'] = c_labels
            s_score = silhouette_score( d[d.columns[1:10]], c_labels )
            ch_score = calinski_harabasz_score( d[d.columns[1:10]], c_labels )
            print( s_score, ch_score )
            
            dct = { 0: 'salmon', 1: 'palegreen', 2: 'skyblue', 3: 'navajowhite', 4: 'crimson', 5: 'seagreen', 6: 'cornflowerblue', 7: 'black', 8: 'slategray' }
            colors = list( map( dct.get, d['cluster_label'] ) )
            fig = plt.figure( figsize = ( 12, 8 ))
            plt.scatter( x_[:,0], x_[:,1], c = colors )
            fig.savefig( 'fig/' + col + '_' + model_key + str(j) + '_TSNE.png', bbox_inches = 'tight' )
            
            p = np.zeros( ( lbl_n, k ) )
            for r in range( lbl_n ):
                for c in range( k ):
                    p[r,c] = d[ ( d['true_label'] == r ) & ( d['cluster_label'] == c ) ].shape[0]
            pp = np.zeros( ( p.shape ) )
            for r in range( lbl_n ):
                for c in range( k ):
                    pp[r,c] = p[r,c] / sum( p[:,c] )
                    
            category_names = ['Case -', 'Case +']
            results = {}
            for k_i in range( k ):
                # results['Cluster ' + str( k_i + 1 )] = np.flipud( pp[:,k_i] )
                results['Cluster ' + str( k_i + 1 )] = pp[:,k_i]
              
            font = {
                'family': 'serif',
                'color':  'darkred',
                'weight': 'bold',
                'size': 16,
                }
            fig, ax = survey( results, category_names, counts )
            plt.title( col, fontsize = 'large', loc = 'left', fontdict = font )
            plt.show()
                
            fig.savefig( 'fig/' + col + '_' + str(j) + '.png', bbox_inches = 'tight' )
            
            # w_file = open( 'text/' + col + '_' + model_key + '_cluster_info.txt', 'w' )
            # w_file.write( col + '\n' )
            
            # barWidth = 0.5
            # r = np.arange( k )
            # fig = plt.figure( 2, figsize=(12,10))
            # plt.bar(r, pp[0,:], color='#ff0000', edgecolor='white', width=barWidth, label = 'case -')
            # plt.bar(r, pp[1,:], bottom=pp[0,:], color='#123456', edgecolor='white', width=barWidth, label = 'case +' )    
            # plt.xlabel( "Cluster labels", fontsize = 'large' )
            # plt.xticks( r )    
            # # le = [ 'negative', 'positive' ]
            # # plt.legend( le, frameon = 1, facecolor = 'white', framealpha = 1 )
            # plt.legend( loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol = 2, fontsize = 'large' )
            # plt.text( 1, .2 , 'Cluster samples: ' + str( counts ), fontsize = 18, backgroundcolor = 'white' )
            # plt.title( col, fontsize = 'x-large' )    
            # # Show graphic
            # plt.show()
            
            xl = pd.DataFrame()
            for ki in range( k ):
                print( "Cluster ", ki + 1 )
                dic = {}
                dic_prop = {}
                dic['Cluster'] = ki + 1
                dic_prop['Cluster'] = 'prop' + str( ki + 1 )
                # w_file.write( "Cluster " + str( ki + 1 ) + '\n' )
                df = d[d['cluster_label'] == ki]
                total = d[d['cluster_label'] == ki].shape[0]
                print( total )
                # w_file.write( "Samples:" + str( total ) + '\n' )
                for icol in range( len( static_var )):
                    if d[static_var[icol]].dtypes == np.object:
                        print( '\n', static_var[icol] )
                        # w_file.write( str( static_var[icol] ) + '\n' )
                        lvl, counts = np.unique( df[static_var[icol]], return_counts = True )
                        for ix in range( len( lvl )):
                            print( lvl[ix], counts[ix], np.round( counts[ix] / total * 100, 2 ), '%' )
                            # w_file.write( str( lvl[ix] ) + ' ' + str( counts[ix] ) + ' ' + str( np.round( counts[ix] / total * 100, 2 ) ) + '%\n' )
                            dic[static_var[icol]+lvl[ix]] = str( counts[ix] )
                            dic_prop[static_var[icol]+lvl[ix]] = np.round( counts[ix] / total * 100, 2 )
                    elif static_var[icol] == 'true_label':
                        print( '\n', static_var[icol] )
                        # w_file.write( str( static_var[icol] ) + '\n' )
                        lvl, counts = np.unique( df[static_var[icol]], return_counts = True )
                        for ix in range( len( lvl )):
                            print( lvl[ix], counts[ix], np.round( counts[ix] / total * 100, 2 ), '%' )
                            # w_file.write( str( lvl[ix] ) + ' ' + str( counts[ix] ) + ' ' + str( np.round( counts[ix] / total * 100, 2 ) ) + '%\n' )
                            dic[static_var[icol]+str(lvl[ix])] = str( counts[ix] )
                            dic_prop[static_var[icol]+str(lvl[ix])] = np.round( counts[ix] / total * 100, 2 )
                    else:
                        print( '\n', static_var[icol] )
                        # w_file.write( str( static_var[icol] ) + '\n' )
                        m = np.round( np.mean( df[static_var[icol]] ), 3 )
                        s = np.round( np.std( df[static_var[icol]] ), 3 )
                        print( 'Mean:', m )
                        print( 'Std:', s )
                        dic[static_var[icol] + 'mean'] = m
                        dic[static_var[icol] + 'std'] = s
                        # w_file.write( 'Mean ' + str( m ) + '\n' )
                        # w_file.write( 'Std ' + str( s ) + '\n' )
                    # w_file.write( '\n' )
                xl = xl.append( dic, ignore_index = True )
                xl = xl.append( dic_prop, ignore_index = True )
            # var = xl.columns
            temp = xl.transpose()
            # xl['var'] = var
            # print( xl )        
            temp.to_csv( 'text/' + col + '_' + model_key + '_cluster_info.csv' )
            # w_file.close()
            # break
            d = d.rename( columns = { 'cluster_label' : model_key + '_label' } )    
            # d.to_csv( 'text/' + col + '_' + model_key + '.csv' )
            del temp
            del xl
        d.to_csv( 'text/' + col + '_clusters.csv' )
        
confidence_level = 0.95
degrees_freedom = sample.shape[0] - 1 
sample_mean = np.mean(sample)
sample_standard_error = sem(sample)

confidence_interval = t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

sheet = 'Sheet'
ss_ = []
for i in range( 1, 9 ):
    ss_.append( pd.read_excel( 'fig/ss.xls', sheet_name=sheet+str(i), header = None))

cl_col = [ 'Obesity', 'Diabetes', 'Cancer', 'BPH', 'Hip Fracture - Osteoporosis', 'Hypertension', 'Alzheimer\'s - Dementia', 'Stroke' ]

def cal_RR( arr, idx = 0 ):
    k = arr.shape[0]
    rr = np.zeros(( k, k ))
    for row in range( k ):
        for col in range( k ):
            rr[row,col] = round( ( arr[col,idx] / sum( arr[col,:] ) ) / ( arr[row,idx] / sum( arr[row,:] ) ), 2 )
            # print( row, col, rr[row,col] )
    # print( rr )
    return rr
    
cl_info = np.empty(( 8 ), dtype = object )
f = open( 'fig/cl_data.txt', 'r' )
test_cases = int( f.readline().strip() )
for t in range( test_cases ):
    runs = int( f.readline().strip() )
    clusters = int( f.readline().strip() )
    classes = 2
    cl = np.zeros( ( clusters, classes, runs ) )
    for i in range( runs ):
        for y in range( clusters ):
            line = f.readline().strip().split()
            cl[y,:,i] = np.array( line, int )
    
    cl_info[t] = cl

conf = np.empty(( 8 ), dtype = object )      
for t in range( test_cases ):
    table = cl_info[t]
    r, c, d = table.shape
    con = np.zeros(( r, c ))
    for row in range( r ):
        for col in range( c ):
            sample = table[row,col,:]
            con[row,col] = 1.96 * np.std( sample ) / np.sqrt( d )
    conf[t] = con
print( conf )

sil_score = [ 0.46, 0.46, 0.45, 0.45, 0.46, 0.44, 0.51, 0.44 ]
ch_score = [ 69317.62, 74958.55, 7135.7, 9127.55, 9127.96, 112875.81, 3752.56, 3305.3 ]

for diag in range( len( cl_col )):
    print( cl_col[diag] )
    data = ss_[diag]
    print( data )
    cl, n_lbl = data.shape
    pp = np.zeros( ( cl, n_lbl ) )
    for r in range( cl ):
        for c in range( n_lbl ):
            pp[r,c] = data.iloc[r,c] / sum( data.iloc[r,:] )
    pp = np.transpose( pp )
    counts = np.sum( data, 1 )
    s_score = sil_score[diag]
    c_score = ch_score[diag]
    con = np.round( conf[diag], 2 )
    
    print( con )
            
    category_names = ['Case -', 'Case +']
    results = {}
    for k_i in range( cl ):
        # results['Cluster ' + str( k_i + 1 )] = np.flipud( pp[:,k_i] )
        results['Cluster ' + str( k_i + 1 )] = pp[:,k_i]
    print( results )
    print( counts )
      
    font = {
        'family': 'serif',
        'color':  'darkred',
        'weight': 'bold',
        'size': 16,
        }
    fig, ax = survey( results, category_names, counts )
    plt.title( cl_col[diag], fontsize = 'large', loc = 'left', fontdict = font )
    plt.show()
    
    fig.savefig( 'fig/' + cl_col[diag] + '_dist.png', bbox_inches = 'tight' )
