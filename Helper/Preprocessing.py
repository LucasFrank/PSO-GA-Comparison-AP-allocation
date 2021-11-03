import pandas as pds
import numpy as npy

class Preprocessing():

    def __init__(self, dfx, t = 5, q = 3, p = 11):
        self.df = dfx.groupby([pds.Grouper(key='time1',freq='{}Min'.format(t))]).agg({'client_id':'count'})
        self.df = self.df.rename(columns={'client_id' : 'count'})
        self.df = self.df.between_time('8:00','20:00')

        self.df['count'] = self.df['count'] + 1

        # Using historic data (Q) from the same time in previous Q-day
        for i in range (1, q + 1):
            self.df['count-{}'.format(i)] = self.df['count'].shift(i * 60 * 24 // t)


        # Change n/a to 1
        self.df = self.df.fillna(1)

        # Normalizing the data
        df_max = max(self.df['count'])
        df_min = min(self.df['count'])
        self.df['count'] = self.df['count'] / (df_max - df_min)
        for i in range (1, q + 1):
            self.df['count-{}'.format(i)] = self.df['count-{}'.format(i)] / (df_max - df_min)


        # Shifiting the data set by Q weeks
        self.df = self.df[q * 60 * 24 // t:]

        #df1 = df1[df1.index.weekday < 5]
        #df1 = df1.between_time('8:00','20:00')
        print('Pre-processing...')

        # Initializing the data
        X1 = list()
        Y1 = list()

        # Mapping each set of variables (P and Q) to their correspondent value
        for i in range(len(self.df) - p - 1):
            X = list()
            for j in range (1, q + 1):
                X.append(self.df['count-{}'.format(j)][i + p + 1])

            X1.append(npy.array(X + list(self.df['count'][i:(i + p)])))
            Y1.append(self.df['count'][i + p + 1])

        self.X = npy.array(X1)
        self.Y = npy.array(Y1)

    def get_data(self):
        return self.X, self.Y
