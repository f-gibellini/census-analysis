import pandas as pd
import numpy as np
import os

def parse_metadata(metadata_string):

    # Dictionary to return
    columns = {}
    
    # right place to split the file
    if '- 50000, 50000+.' in metadata_string:
        column_section = metadata_string.split('- 50000, 50000+.')[1].strip()
    else:
        print('Metadata format is unexpected, cannot find column values definition', line)
        return {}

    for line in column_section.split('.'):
        try:
            #key values for my mapping dict
            k,v = line.strip().replace('|', '').split(':')
            
            if v.strip().lower() == 'continuous':
                columns[k] = set()
            else:
                columns[k] = set([val.strip().lower() for val in v.split(',')])
        
        except:
            print('Couldnt parse line', line)
            
        
    return columns

def map_columns(df, columns_map):
    mapped_df_cols = set()
    mapped_meta_cols = set()
    column_mapping = {}

    #try to map columns based on expected values
    for meta_col, value_set in columns_map.items():

        #when empty set means continous column, skip for now, will map later
        if len(value_set) == 0:
            continue

        #initialize values to keep track
        best_match = None
        best_score = 0

        #go through all DF columns
        for col in df.columns:
            #if current mapped already continue
            if col in mapped_df_cols:
                continue
                
            actual_values = set(str(val) for val in df[col].unique())
                            
            common_values = actual_values.intersection(value_set)

            #score based on how much overlap there is
            score = len(common_values) / max(len(actual_values), 1)

            #keep track of score
            if score > best_score:
                best_score = score
                best_match = col
                
        if best_match:
            column_mapping[best_match] = meta_col
            mapped_df_cols.add(best_match)
            mapped_meta_cols.add(meta_col)

    #now map remaining columns, assuming they are the continuos ones
    cont_meta_cols = [col for col, v in columns_map.items() if len(v) == 0]
    cont_df_cols = [col for col in df.columns if (not (col in mapped_df_cols))]
    
    for i, df_col in enumerate(cont_df_cols):
        if i < len(cont_meta_cols):
            column_mapping[df_col] = cont_meta_cols[i]
    
    return column_mapping



def read_data(fname, 
              metadata_fname,
              target_name = 'income'):

    if os.path.isfile(fname):        
        df = pd.read_csv(fname,
                      names = [f'col_{i}' for i in range(42)])
    else:
        print('CSV File not Found')
        return -1

    print('Read input data file', fname)
    #make sure all strings are lowercase and stripped
    df = df.applymap(lambda x:x.lower().strip() if (isinstance(x, str)) else (x))

    #read metadata file
    if os.path.isfile(metadata_fname):
        with open(metadata_fname, 'r') as f:
            mdata_string = f.read()
    else:
        print('Metadata File not Found')
        return -1

    print('Read metadata file', metadata_fname)
    
    #create dict to map columns based on metadata values
    columns_map = parse_metadata(mdata_string)
    
    #create columns map and map columns in df, assume last column is target (income > o < 50k)
    df_cols_map = map_columns(df, columns_map)
    df = df.rename(columns = {'col_41': target_name ,**df_cols_map})
    
    print('Mapped columns')
    # Map all string containing 'Not in universe' to NULL
    df = df.applymap(lambda x:np.nan if (isinstance(x, str) and (('not in universe' in x) or ('?' in x) or (x == 'na')
                                                                            )) else (x))
    
    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    return df
