# import the required libraries
import pandas as pd

# define the months
files = ['_201702', '_201701', '_201612', '_201611', '_201610','_201609']

# main function
def main():

    user_logs, extension = 'user_logs', '.csv'

    table = pd.DataFrame(index = ['msno'])                                                         # define an empty dataframe with index as 'msno'

    for i in files:
    
        df = pd.read_csv(user_logs + i + extension)                                                # read the .csv files
    
        # aggregate and calculate the mean, standard deviation, and count

        df = df.groupby('msno').agg({'num_25':['mean','std'], 'num_50':['mean','std'], 
                                    'num_75':['mean','std'], 'num_985':['mean','std'], 
                                    'num_100':['mean','std'], 'num_unq':['mean','std'], 
                                    'total_secs':['mean','std'], 'date':'count'}).reset_index()
    
        # assign the column names

        df.columns = ['msno', 'num_25_mean', 'num_25_std', 'num_50_mean', 
                      'num_50_std', 'num_75_mean', 'num_75_std', 'num_985_mean', 
                      'num_985_std', 'num_100_mean', 'num_100_std', 'num_unq_mean', 
                      'num_unq_std', 'total_secs_mean', 'total_secs_std', 'count']
    
        df.columns = ['msno'] + [str(col) + i for col in df.columns[1:]]                          # rename the columns
    
        table = table.join(df.set_index('msno'), how = 'outer')                                   # accumulate the results

    table.fillna(0, inplace = True)                                                               # replace the 'NaN' with 0

    table.to_csv('user_logs_initial.csv')                                                         # write the output file

# main function check

if __name__ == '__main__':
    main()