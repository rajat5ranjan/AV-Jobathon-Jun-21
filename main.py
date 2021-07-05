# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import warnings
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. .read_csv)
from scipy import stats
from datetime import datetime
from tqdm import tqdm_notebook, tqdm

tqdm.pandas(desc="Running Date Time conversion")
warnings.filterwarnings("ignore")


class ETLPipeline(object):
    def __init__(self):
        self.visitorData = None
        self.userData = None
        self.visUsrData = None
        self.df = None

    def extract(self):
        """ Extract and load the data """
        self.visitorData = pd.read_csv("data/VisitorLogsData.csv",
                                       usecols=['webClientID', 'VisitDateTime', 'ProductID', 'UserID', 'Activity',
                                                'OS'])
        self.userData = pd.read_csv("data/userTable.csv")

    def feature_engineering(self):
        """ Feature engineering and imputing missing data"""
        self.visUsrData['Activity'] = self.visUsrData.sort_values(['UserID', 'VisitDateTime']).groupby('UserID')[
            'Activity'].transform(lambda x: x.fillna(method='bfill'))
        self.visUsrData['Activity'].fillna('pageload', inplace=True)

        self.visUsrData['OS'] = self.visUsrData['OS'].apply(lambda x: x.lower())
        self.visUsrData['ProductID'] = self.visUsrData['ProductID'].apply(lambda x: x if x != x else str(x).lower())
        self.visUsrData['Activity'] = self.visUsrData['Activity'].apply(lambda x: x if x != x else str(x).lower())

        self.visUsrData['SevenDays'] = 0
        self.visUsrData.loc[self.visUsrData['VisitDateTime'] >= '2018-05-21', 'SevenDays'] = 1
        self.visUsrData['FifteenDays'] = 0
        self.visUsrData.loc[self.visUsrData['VisitDateTime'] >= '2018-05-13', 'FifteenDays'] = 1
        self.visUsrData['isActive'] = self.visUsrData['Activity'].apply(lambda x: x == x)

        self.visUsrData['is7Active'] = 0
        self.visUsrData.loc[(self.visUsrData['isActive'] == 1) & (self.visUsrData['SevenDays'] == 1), 'is7Active'] = 1

        self.visUsrData['VisitDate'] = self.visUsrData['VisitDateTime'].dt.date

        self.visUsrData['Pageloads_last_7_days'] = 0
        self.visUsrData.loc[
            (self.visUsrData['Activity'] == 'pageload') & (
                    self.visUsrData['SevenDays'] == 1), 'Pageloads_last_7_days'] = 1

        self.visUsrData['Clicks_last_7_days'] = 0
        self.visUsrData.loc[
            (self.visUsrData['Activity'] == 'click') & (self.visUsrData['SevenDays'] == 1), 'Clicks_last_7_days'] = 1

        self.visUsrData['FifteenDaysActive'] = 0
        self.visUsrData.loc[
            ((self.visUsrData['FifteenDays'] == 1) & (self.visUsrData['isActive'] == True)), 'FifteenDaysActive'] = 1
        self.visUsrData['pageloads_actvity'] = 0
        self.visUsrData.loc[(self.visUsrData['Activity'] == 'pageload'), 'pageloads_actvity'] = 1
        self.visUsrData['ProductID'] = self.visUsrData.sort_values(['UserID', 'VisitDateTime']).groupby('UserID')[
            'ProductID'].transform(lambda x: x.fillna(method='bfill'))

    def transform(self):
        """Data wrangling and transformation for the input features and returns the results"""
        self.visUsrData = self.visitorData[self.visitorData['UserID'].isnull() == False].copy()
        self.visUsrData['VisitDateTime'] = self.visUsrData['VisitDateTime'].apply(correct_timestamp)

        # drop duplicates
        self.visUsrData = self.visUsrData[self.visUsrData.duplicated() == False].copy()
        self.visUsrData['VisitDateTime'] = self.visUsrData.groupby(['UserID', 'webClientID'])[
            'VisitDateTime'].transform(
            lambda x: x.fillna(x.min()))
        print("\t \t Feature Engineering")
        self.feature_engineering()

        self.df = pd.DataFrame(
            self.visUsrData.groupby(['UserID'])['webClientID'].count().reset_index().drop('webClientID', axis=1))
        # No_of_days_Visited_7_Days
        print("\t \t \t No_of_days_Visited_7_Days")
        self.df = self.df.merge(
            self.visUsrData.groupby(['UserID', 'VisitDate', 'is7Active']).count().reset_index().groupby('UserID')[
                'is7Active'].sum().reset_index(), on='UserID', how='left')
        self.df['is7Active'].fillna(0, inplace=True)
        # merging user data and User_Vintage
        print("\t \t \t User_Vintage")
        self.visUsrData = self.visUsrData.merge(self.userData, on='UserID', how='inner')
        self.visUsrData['Signup Date'] = pd.to_datetime(self.visUsrData['Signup Date'], format="%Y-%m-%d %H:%M:%S")
        self.visUsrData['Signup Date'] = self.visUsrData['Signup Date'].dt.tz_localize(None)
        self.visUsrData['User_Vintage'] = (
                self.visUsrData['VisitDateTime'].max() - self.visUsrData['Signup Date']).dt.days

        self.df = self.df.merge(self.visUsrData.groupby(['UserID'])['User_Vintage'].max().reset_index(), on='UserID',
                                how='left')
        self.df['User_Vintage'] = self.df['User_Vintage'] + 1
        # Most_Active_OS
        print("\t \t \t Most_Active_OS")
        self.df['Most_Active_OS'] = self.visUsrData.groupby(['UserID'])['OS'].agg(lambda x: stats.mode(x)[0][0]).values

        # Pageloads_last_7_days & Clicks_last_7_days
        print("\t \t \t Pageloads_last_7_days & Clicks_last_7_days")
        self.df['Pageloads_last_7_days'] = self.visUsrData.groupby(['UserID'])['Pageloads_last_7_days'].sum().values
        self.df['Clicks_last_7_days'] = self.visUsrData.groupby(['UserID'])['Clicks_last_7_days'].sum().values

        #
        print("\t \t \t Recently_Viewed_Product")
        mask = self.visUsrData['pageloads_actvity'] == 1
        self.df = self.df.merge(self.visUsrData.sort_values(['UserID', 'VisitDateTime'])[mask].groupby(['UserID'])
            .agg({'ProductID': 'last'}).reset_index().rename(
            columns={'ProductID': 'Recently_Viewed_Product'}), on='UserID', how='left')

        print("\t \t \t Most_Viewed_product_15_Days")
        mask2 = (self.visUsrData['pageloads_actvity'] == 1) & (self.visUsrData['FifteenDays'] == 1)
        self.df = self.df.merge(
            self.visUsrData.sort_values(['UserID', 'VisitDateTime'], ascending=[True, False])[mask2].groupby(['UserID'])
                .agg({'ProductID': lambda x: stats.mode(x)[0][0]}).reset_index().rename(
                columns={'ProductID': 'Most_Viewed_product_15_Days'}), on='UserID', how='left')

        print("\t \t \t No_Of_Products_Viewed_15_Days")
        self.visUsrData['ProductID'] = self.visUsrData.groupby(['UserID'])['ProductID'].transform(
            lambda x: x.fillna(stats.mode(x)[0][0]))
        self.visUsrData['ProductID'].fillna('Product101', inplace=True)
        mask3 = self.visUsrData['FifteenDays'] == 1

        self.df = self.df.merge(
            self.visUsrData[mask3].groupby(['UserID']).agg({'ProductID': 'nunique'}).reset_index().rename(
                columns={'ProductID': 'No_Of_Products_Viewed_15_Days'}), on='UserID', how='left')

    def load(self):
        """Loads the input features and returns the results"""
        self.df.reset_index(inplace=True)
        self.df.rename(columns={'is7Active': 'No_of_days_Visited_7_Days'}, inplace=True)
        self.df['Recently_Viewed_Product'].fillna('Product101', inplace=True)
        self.df['Most_Viewed_product_15_Days'].fillna('Product101', inplace=True)
        self.df['No_Of_Products_Viewed_15_Days'].fillna(0, inplace=True)
        self.df = self.df.reindex(
            ['UserID', 'No_of_days_Visited_7_Days', 'No_Of_Products_Viewed_15_Days', 'User_Vintage',
             'Most_Viewed_product_15_Days',
             'Most_Active_OS', 'Recently_Viewed_Product', 'Pageloads_last_7_days', 'Clicks_last_7_days'],
            axis=1)
        self.df.to_csv('data/output/input_feats.csv', index=False)


def correct_timestamp(x):
    """Date Time conversion handling"""
    if x == x:
        try:
            x = pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            x = int(x)
            x = x / 10 ** 9
            x = datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f')
        return x
    else:
        return np.nan


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_time = datetime.now()
    pipeline = ETLPipeline()
    print('Data Pipeline created')
    print('\t extracting data from source .... ')
    pipeline.extract()
    print('\t formatting and transforming data ... ')
    pipeline.transform()
    print('\t loading into CSV ... ')
    pipeline.load()
    print(datetime.now() - start_time)
