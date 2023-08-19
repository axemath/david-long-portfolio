# Titanic

This script is entered in the [Titanic](https://www.kaggle.com/competitions/titanic/overview) competition on Kaggle.  As of 08/19/23, it was 227/14,944 on the Leaderboard under the name 'axemath'.  Given a training set of data with variables such as Passenger Class, Fare, and Gender, the objective is to predict the survival of passengers in the test set.  The metric is accuracy.

After data cleansing and feature engineering, Logistic Regression, Support Vector Machine, and Random Forest models are fed to a Voting Classifier which renders the predictions.  The Sci-Kit Learn API is used.


```python
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
```

### Set some options, import the data, summarize data types and missing values


```python
# set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 25)

# local directory
path = 'C:\\Users\\Zen Warrior\\OneDrive\\kaggle\\'

# head, shape and data types
X = pd.read_csv(path +'titanic\\train.csv')
X_t = pd.read_csv(path +'titanic\\test.csv')
print('\n\nX.head():\n', X.head(), '\n\n', sep='')
print('X_t.head():\n', X_t.head(), '\n\n', sep='')
print('Size of X: ', X.shape, '\n\n')
print('Size of X_t: ', X_t.shape, '\n\n')
print('X Data Types:\n', X.dtypes, '\n\n', sep='')
print('X_t Data Types:\n', X_t.dtypes, '\n\n', sep='')

# summary of missing values
print('Summary of Missing Values in X:\n', X.isnull().sum(), '\n\n', sep='')
print('Summary of Missing Values in X_t:\n', 
      X_t.isnull().sum(), '\n\n', sep='')
```

    
    
    X.head():
       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    
    
    X_t.head():
       PassengerId  Pclass                                          Name     Sex  \
    0          892       3                              Kelly, Mr. James    male   
    1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   
    2          894       2                     Myles, Mr. Thomas Francis    male   
    3          895       3                              Wirz, Mr. Albert    male   
    4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   
    
        Age  SibSp  Parch   Ticket     Fare Cabin Embarked  
    0  34.5      0      0   330911   7.8292   NaN        Q  
    1  47.0      1      0   363272   7.0000   NaN        S  
    2  62.0      0      0   240276   9.6875   NaN        Q  
    3  27.0      0      0   315154   8.6625   NaN        S  
    4  22.0      1      1  3101298  12.2875   NaN        S  
    
    
    Size of X:  (891, 12) 
    
    
    Size of X_t:  (418, 11) 
    
    
    X Data Types:
    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object
    
    
    X_t Data Types:
    PassengerId      int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object
    
    
    Summary of Missing Values in X:
    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    
    
    Summary of Missing Values in X_t:
    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64
    
    
    

### Split data by dtype


```python
# split data frames by dtype
y_train = np.array(X.pop('Survived'))
X_num = X.loc[:,['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']]
X_cat = X.loc[:,['Embarked', 'Cabin']]

X_t_num = X_t.loc[:,['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']]
X_t_cat = X_t.loc[:,['Embarked', 'Cabin']]
```

### Transform Sex

I thought it might be useful to add a category that indicates whether a female is married.  This is accomplished with a customized Sci-Kit Learn class and a regular expression.


```python
# TransformSex
married_female_re = 'Mrs'
class TransformSex(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gender = []
        self.name = []
        self.sex = []
    def fit(self, Z):
        return self
    def transform(self, Z):
        gender = Z['Sex']
        name = Z['Name']
        sex = []
        for i, g in enumerate(gender):
            if (g == 'female') and (re.search(married_female_re, name[i])):
                sex.append('married_female')
            else:
                sex.append(g)
        return sex
transform_sex = TransformSex()
X_cat['Sex'] = transform_sex.transform(X.loc[:,['Sex', 'Name']])
X_t_cat['Sex'] = transform_sex.transform(X_t.loc[:,['Sex', 'Name']])
```

### Transform Name

Last name is used as a feature in place of full name.  It's less sparse, and likely a better indicator of survival, as family tends to stick together.


```python
# TransformName
class TransformName(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.last_names = []
    def fit(self, Z):
        return self
    def transform(self, Z):
        last_names = []
        name_split = Z.str.split(pat=',')
        for index, split in enumerate(name_split):
            last_names.append(split[0])
        return last_names
transform_name = TransformName()
X_cat['LastName'] = transform_name.transform(X['Name'])
X_t_cat['LastName'] = transform_name.transform(X_t['Name'])
```

### Transform Ticket

After inspecting the Ticket values, it was apparent that some cleansing was necessary.  For example, "A./4." is likely the same category as "A/4".  This transformation makes some assumptions, but isn't overly agressive in making them.  Some tickets have an alphanumeric prefix, others are just numbers.  The ticket was split into two features, one called `ticket_word`, which contains the alphanumeric prefix, and `ticket_prefix`, which contains the first three digits of the ticket number.  


```python
# TransformTicket
class TransformTicket(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ticket_word = []
        self.ticket_prefix = []
    def fit(self, Z):
        return self
    def transform(self, Z):
        ticket_word = []
        ticket_prefix = []
        Z = Z.str.upper()
        Z = Z.str.replace('.', '')
        Z = Z.str.replace('/', '')
        Z = Z.str.replace('STONO ', 'STONO')
        Z = Z.str.replace('STONO', 'SOTONO')
        Z = Z.str.replace('SOPP', 'SOP')
        for index, split in enumerate(Z.str.rsplit(n=1)):
            if len(split) > 1:
                ticket_word.append(split[0])
                ticket_prefix.append(split[1][0:3])
            elif split[0] != 'LINE':
                ticket_word.append(np.nan)
                ticket_prefix.append(split[0][0:3])
            else:
                ticket_word.append(split[0])
                ticket_prefix.append('0')
        return ticket_word, ticket_prefix    
transform_ticket = TransformTicket()
X_cat['TicketWord'], X_cat['TicketPrefix'] =\
    transform_ticket.transform(X['Ticket'])
X_t_cat['TicketWord'], X_t_cat['TicketPrefix'] =\
    transform_ticket.transform(X_t['Ticket'])
```

### Unique Values


```python
# unique values
for col in X_num:
    print('\nSeries Name: ', col)
    print('Series Type: ', X_num[col].dtype)
    print('Number of Unique Values X_num: ', pd.unique(X_num[col]).shape[0])
    print('Number of Unique Values X_t_num: ',
          pd.unique(X_t_num[col]).shape[0])
    print('Unique Values X_num:\n', np.sort(pd.unique(X_num[col])), sep='')
    print('Unique Values X_t_num:\n', np.sort(pd.unique(X_t_num[col])), 
          sep='')   

for col in X_cat:
    print('\nSeries Name: ', col)
    print('Series Type: ', X_cat[col].dtype)
    print('Number of Unique Values X_cat: ', pd.unique(X_cat[col]).shape[0])
    print('Number of Unique Values X_t_cat: ', 
          pd.unique(X_t_cat[col]).shape[0])
    if col not in ['LastName']:
        print('Unique Values X_cat:\n',
              pd.unique(X_cat[col].sort_values()), sep='')
        print('Unique Values X_t_cat:\n', 
              pd.unique(X_t_cat[col].sort_values()), sep='')
```

    
    Series Name:  Pclass
    Series Type:  int64
    Number of Unique Values X_num:  3
    Number of Unique Values X_t_num:  3
    Unique Values X_num:
    [1 2 3]
    Unique Values X_t_num:
    [1 2 3]
    
    Series Name:  SibSp
    Series Type:  int64
    Number of Unique Values X_num:  7
    Number of Unique Values X_t_num:  7
    Unique Values X_num:
    [0 1 2 3 4 5 8]
    Unique Values X_t_num:
    [0 1 2 3 4 5 8]
    
    Series Name:  Parch
    Series Type:  int64
    Number of Unique Values X_num:  7
    Number of Unique Values X_t_num:  8
    Unique Values X_num:
    [0 1 2 3 4 5 6]
    Unique Values X_t_num:
    [0 1 2 3 4 5 6 9]
    
    Series Name:  Age
    Series Type:  float64
    Number of Unique Values X_num:  89
    Number of Unique Values X_t_num:  80
    Unique Values X_num:
    [ 0.42  0.67  0.75  0.83  0.92  1.    2.    3.    4.    5.    6.    7.
      8.    9.   10.   11.   12.   13.   14.   14.5  15.   16.   17.   18.
     19.   20.   20.5  21.   22.   23.   23.5  24.   24.5  25.   26.   27.
     28.   28.5  29.   30.   30.5  31.   32.   32.5  33.   34.   34.5  35.
     36.   36.5  37.   38.   39.   40.   40.5  41.   42.   43.   44.   45.
     45.5  46.   47.   48.   49.   50.   51.   52.   53.   54.   55.   55.5
     56.   57.   58.   59.   60.   61.   62.   63.   64.   65.   66.   70.
     70.5  71.   74.   80.     nan]
    Unique Values X_t_num:
    [ 0.17  0.33  0.75  0.83  0.92  1.    2.    3.    5.    6.    7.    8.
      9.   10.   11.5  12.   13.   14.   14.5  15.   16.   17.   18.   18.5
     19.   20.   21.   22.   22.5  23.   24.   25.   26.   26.5  27.   28.
     28.5  29.   30.   31.   32.   32.5  33.   34.   34.5  35.   36.   36.5
     37.   38.   38.5  39.   40.   40.5  41.   42.   43.   44.   45.   46.
     47.   48.   49.   50.   51.   53.   54.   55.   57.   58.   59.   60.
     60.5  61.   62.   63.   64.   67.   76.     nan]
    
    Series Name:  Fare
    Series Type:  float64
    Number of Unique Values X_num:  248
    Number of Unique Values X_t_num:  170
    Unique Values X_num:
    [  0.       4.0125   5.       6.2375   6.4375   6.45     6.4958   6.75
       6.8583   6.95     6.975    7.0458   7.05     7.0542   7.125    7.1417
       7.225    7.2292   7.25     7.3125   7.4958   7.5208   7.55     7.6292
       7.65     7.725    7.7292   7.7333   7.7375   7.7417   7.75     7.775
       7.7875   7.7958   7.8      7.8292   7.8542   7.875    7.8792   7.8875
       7.8958   7.925    8.0292   8.05     8.1125   8.1375   8.1583   8.3
       8.3625   8.4042   8.4333   8.4583   8.5167   8.6542   8.6625   8.6833
       8.7125   8.85     9.       9.2167   9.225    9.35     9.475    9.4833
       9.5      9.5875   9.825    9.8375   9.8417   9.8458  10.1708  10.4625
      10.5     10.5167  11.1333  11.2417  11.5     12.      12.275   12.2875
      12.35    12.475   12.525   12.65    12.875   13.      13.4167  13.5
      13.7917  13.8583  13.8625  14.      14.1083  14.4     14.4542  14.4583
      14.5     15.      15.0458  15.05    15.1     15.2458  15.5     15.55
      15.7417  15.75    15.85    15.9     16.      16.1     16.7     17.4
      17.8     18.      18.75    18.7875  19.2583  19.5     19.9667  20.2125
      20.25    20.525   20.575   21.      21.075   21.6792  22.025   22.3583
      22.525   23.      23.25    23.45    24.      24.15    25.4667  25.5875
      25.925   25.9292  26.      26.25    26.2833  26.2875  26.3875  26.55
      27.      27.7208  27.75    27.9     28.5     28.7125  29.      29.125
      29.7     30.      30.0708  30.5     30.6958  31.      31.275   31.3875
      32.3208  32.5     33.      33.5     34.0208  34.375   34.6542  35.
      35.5     36.75    37.0042  38.5     39.      39.4     39.6     39.6875
      40.125   41.5792  42.4     46.9     47.1     49.5     49.5042  50.
      50.4958  51.4792  51.8625  52.      52.5542  53.1     55.      55.4417
      55.9     56.4958  56.9292  57.      57.9792  59.4     61.175   61.3792
      61.9792  63.3583  65.      66.6     69.3     69.55    71.      71.2833
      73.5     75.25    76.2917  76.7292  77.2875  77.9583  78.2667  78.85
      79.2     79.65    80.      81.8583  82.1708  83.1583  83.475   86.5
      89.1042  90.      91.0792  93.5    106.425  108.9    110.8833 113.275
     120.     133.65   134.5    135.6333 146.5208 151.55   153.4625 164.8667
     211.3375 211.5    221.7792 227.525  247.5208 262.375  263.     512.3292]
    Unique Values X_t_num:
    [  0.       3.1708   6.4375   6.4958   6.95     7.       7.05     7.225
       7.2292   7.25     7.2833   7.55     7.575    7.5792   7.6292   7.65
       7.7208   7.725    7.7333   7.75     7.775    7.7792   7.7958   7.8208
       7.8292   7.85     7.8542   7.8792   7.8875   7.8958   7.925    8.05
       8.1125   8.5167   8.6625   8.7125   8.9625   9.225    9.325    9.35
       9.5      9.6875  10.5     10.7083  11.5     12.1833  12.2875  12.35
      12.7375  12.875   13.      13.4167  13.5     13.775   13.8583  13.8625
      13.9     14.1083  14.4     14.4542  14.4583  14.5     15.0333  15.0458
      15.1     15.2458  15.5     15.55    15.5792  15.7417  15.75    15.9
      16.      16.1     16.7     17.4     18.      20.2125  20.25    20.575
      21.      21.075   21.6792  22.025   22.3583  22.525   23.      23.25
      23.45    24.15    25.4667  25.7     25.7417  26.      26.55    27.4458
      27.7208  27.75    28.5     28.5375  29.      29.125   29.7     30.
      30.5     31.3875  31.5     31.6792  31.6833  32.5     34.375   36.75
      37.0042  39.      39.4     39.6     39.6875  41.5792  42.4     42.5
      45.5     46.9     47.1     50.      50.4958  51.4792  51.8625  52.
      52.5542  53.1     55.4417  56.4958  57.75    59.4     60.      61.175
      61.3792  61.9792  63.3583  65.      69.55    71.2833  73.5     75.2417
      75.25    76.2917  78.85    79.2     81.8583  82.2667  83.1583  90.
      93.5    106.425  108.9    134.5    135.6333 136.7792 146.5208 151.55
     164.8667 211.3375 211.5    221.7792 227.525  247.5208 262.375  263.
     512.3292      nan]
    
    Series Name:  Embarked
    Series Type:  object
    Number of Unique Values X_cat:  4
    Number of Unique Values X_t_cat:  3
    Unique Values X_cat:
    ['C' 'Q' 'S' nan]
    Unique Values X_t_cat:
    ['C' 'Q' 'S']
    
    Series Name:  Cabin
    Series Type:  object
    Number of Unique Values X_cat:  148
    Number of Unique Values X_t_cat:  77
    Unique Values X_cat:
    ['A10' 'A14' 'A16' 'A19' 'A20' 'A23' 'A24' 'A26' 'A31' 'A32' 'A34' 'A36'
     'A5' 'A6' 'A7' 'B101' 'B102' 'B18' 'B19' 'B20' 'B22' 'B28' 'B3' 'B30'
     'B35' 'B37' 'B38' 'B39' 'B4' 'B41' 'B42' 'B49' 'B5' 'B50' 'B51 B53 B55'
     'B57 B59 B63 B66' 'B58 B60' 'B69' 'B71' 'B73' 'B77' 'B78' 'B79' 'B80'
     'B82 B84' 'B86' 'B94' 'B96 B98' 'C101' 'C103' 'C104' 'C106' 'C110' 'C111'
     'C118' 'C123' 'C124' 'C125' 'C126' 'C128' 'C148' 'C2' 'C22 C26'
     'C23 C25 C27' 'C30' 'C32' 'C45' 'C46' 'C47' 'C49' 'C50' 'C52' 'C54'
     'C62 C64' 'C65' 'C68' 'C7' 'C70' 'C78' 'C82' 'C83' 'C85' 'C86' 'C87'
     'C90' 'C91' 'C92' 'C93' 'C95' 'C99' 'D' 'D10 D12' 'D11' 'D15' 'D17' 'D19'
     'D20' 'D21' 'D26' 'D28' 'D30' 'D33' 'D35' 'D36' 'D37' 'D45' 'D46' 'D47'
     'D48' 'D49' 'D50' 'D56' 'D6' 'D7' 'D9' 'E10' 'E101' 'E12' 'E121' 'E17'
     'E24' 'E25' 'E31' 'E33' 'E34' 'E36' 'E38' 'E40' 'E44' 'E46' 'E49' 'E50'
     'E58' 'E63' 'E67' 'E68' 'E77' 'E8' 'F E69' 'F G63' 'F G73' 'F2' 'F33'
     'F38' 'F4' 'G6' 'T' nan]
    Unique Values X_t_cat:
    ['A11' 'A18' 'A21' 'A29' 'A34' 'A9' 'B10' 'B11' 'B24' 'B26' 'B36' 'B41'
     'B45' 'B51 B53 B55' 'B52 B54 B56' 'B57 B59 B63 B66' 'B58 B60' 'B61' 'B69'
     'B71' 'B78' 'C101' 'C105' 'C106' 'C116' 'C130' 'C132' 'C22 C26'
     'C23 C25 C27' 'C28' 'C31' 'C32' 'C39' 'C46' 'C51' 'C53' 'C54' 'C55 C57'
     'C6' 'C62 C64' 'C7' 'C78' 'C80' 'C85' 'C86' 'C89' 'C97' 'D' 'D10 D12'
     'D15' 'D19' 'D21' 'D22' 'D28' 'D30' 'D34' 'D37' 'D38' 'D40' 'D43' 'E31'
     'E34' 'E39 E41' 'E45' 'E46' 'E50' 'E52' 'E60' 'F' 'F E46' 'F E57' 'F G63'
     'F2' 'F33' 'F4' 'G6' nan]
    
    Series Name:  Sex
    Series Type:  object
    Number of Unique Values X_cat:  3
    Number of Unique Values X_t_cat:  3
    Unique Values X_cat:
    ['female' 'male' 'married_female']
    Unique Values X_t_cat:
    ['female' 'male' 'married_female']
    
    Series Name:  LastName
    Series Type:  object
    Number of Unique Values X_cat:  667
    Number of Unique Values X_t_cat:  352
    
    Series Name:  TicketWord
    Series Type:  object
    Number of Unique Values X_cat:  28
    Number of Unique Values X_t_cat:  24
    Unique Values X_cat:
    ['A4' 'A5' 'AS' 'C' 'CA' 'CASOTON' 'FA' 'FC' 'FCC' 'LINE' 'PC' 'PP' 'PPP'
     'SC' 'SCA4' 'SCAH' 'SCAH BASLE' 'SCOW' 'SCPARIS' 'SOC' 'SOP' 'SOTONO2'
     'SOTONOQ' 'SP' 'SWPP' 'WC' 'WEP' nan]
    Unique Values X_t_cat:
    ['A 2' 'A4' 'A5' 'AQ3' 'AQ4' 'C' 'CA' 'FC' 'FCC' 'LP' 'PC' 'PP' 'SC'
     'SCA3' 'SCA4' 'SCAH' 'SCPARIS' 'SOC' 'SOP' 'SOTONO2' 'SOTONOQ' 'WC' 'WEP'
     nan]
    
    Series Name:  TicketPrefix
    Series Type:  object
    Number of Unique Values X_cat:  170
    Number of Unique Values X_t_cat:  132
    Unique Values X_cat:
    ['0' '104' '110' '111' '112' '113' '116' '117' '118' '119' '122' '124'
     '127' '130' '132' '135' '142' '143' '148' '149' '158' '160' '169' '172'
     '173' '174' '175' '176' '177' '185' '187' '198' '199' '200' '205' '207'
     '211' '212' '213' '214' '215' '216' '218' '219' '220' '222' '223' '226'
     '228' '229' '230' '231' '233' '234' '235' '236' '237' '239' '240' '241'
     '243' '244' '245' '246' '248' '250' '262' '263' '264' '265' '266' '267'
     '268' '269' '270' '272' '278' '281' '282' '284' '285' '286' '290' '291'
     '292' '293' '295' '297' '3' '308' '310' '312' '314' '315' '319' '323'
     '324' '330' '331' '333' '334' '335' '336' '338' '340' '341' '342' '343'
     '345' '346' '347' '348' '349' '350' '352' '353' '354' '358' '359' '362'
     '363' '364' '365' '367' '368' '369' '370' '371' '372' '373' '374' '376'
     '382' '383' '384' '386' '390' '392' '394' '398' '400' '413' '434' '453'
     '457' '488' '541' '545' '546' '554' '572' '573' '621' '653' '656' '660'
     '693' '695' '707' '726' '751' '752' '753' '754' '755' '759' '847' '851'
     '923' '954']
    Unique Values X_t_cat:
    ['110' '111' '112' '113' '117' '118' '122' '127' '129' '130' '132' '135'
     '136' '139' '142' '147' '148' '151' '158' '160' '169' '173' '174' '175'
     '176' '177' '198' '199' '2' '200' '207' '211' '212' '213' '214' '215'
     '216' '220' '226' '228' '230' '231' '233' '234' '235' '236' '237' '239'
     '240' '241' '242' '244' '248' '250' '251' '254' '262' '263' '265' '266'
     '267' '268' '269' '280' '281' '282' '284' '286' '290' '291' '292' '297'
     '306' '307' '308' '310' '313' '314' '315' '323' '329' '330' '331' '333'
     '334' '335' '336' '340' '341' '342' '343' '345' '346' '347' '348' '349'
     '350' '359' '363' '364' '365' '366' '367' '368' '369' '370' '371' '376'
     '382' '383' '386' '391' '392' '400' '413' '427' '488' '498' '573' '621'
     '653' '660' '680' '694' '726' '752' '753' '754' '793' '851' '923' '954']
    

### Histograms of Continuous Data

Fare is clearly positively skewed.  I attempted a log transformation, but it didn't make a difference in the accuracy of the model, so I left it as is in the final run.  Age is close enough to symmetric and mound-shaped.


```python
plt.hist(X['Fare'])
```




    (array([732., 106.,  31.,   2.,  11.,   6.,   0.,   0.,   0.,   3.]),
     array([  0.     ,  51.23292, 102.46584, 153.69876, 204.93168, 256.1646 ,
            307.39752, 358.63044, 409.86336, 461.09628, 512.3292 ]),
     <BarContainer object of 10 artists>)




    
![png](output_15_1.png)
    



```python
plt.hist(X['Age'])
```




    (array([ 54.,  46., 177., 169., 118.,  70.,  45.,  24.,   9.,   2.]),
     array([ 0.42 ,  8.378, 16.336, 24.294, 32.252, 40.21 , 48.168, 56.126,
            64.084, 72.042, 80.   ]),
     <BarContainer object of 10 artists>)




    
![png](output_16_1.png)
    


### Impute missing values

`LastName`, `TicketWord`, and `Cabin` are imputed with 'missing_value'.  `TicketPrefix` is imputed with '0'.  The variables `Sex`, `Embarked`, `Pclass`, `SibSp`, and `Parch` are imputed with their modes, while `Age` and `Fare` are imputed with their means.  By using Sci-Kit Learn's `SimpleImputer()`, it's assured that variables with missing values in the test set are imputed correctly, even if there are no missing values for the variables in the training set.


```python
# missing_imputer
missing_imputer = SimpleImputer(strategy='constant')
missing_imputer.fit(
    np.reshape(
        X_cat.loc[:,X_cat.columns.isin(['LastName', 'TicketWord', 'Cabin'])],\
        (-1, 3)
    )
)
X_cat.loc[:,X_cat.columns.isin(['LastName', 'TicketWord', 'Cabin'])] = \
    missing_imputer.transform(np.reshape(
        X_cat.loc[:,X_cat.columns.isin(['LastName', 'TicketWord', 'Cabin'])],\
        (-1, 3))
    )
X_t_cat.loc[:,X_t_cat.columns.isin(['LastName', 'TicketWord', 'Cabin'])] = \
    missing_imputer.transform(np.reshape(
        X_t_cat.loc[:,X_t_cat.columns.isin(
            ['LastName', 'TicketWord', 'Cabin'])],\
        (-1, 3))
    )        
print('\n\nmissing_imputer values: ', missing_imputer.statistics_, '\n\n')

# ticket_prefix_imputer
ticket_prefix_imputer = SimpleImputer(strategy='constant', fill_value='0')
ticket_prefix_imputer.fit(np.reshape(X_cat['TicketPrefix'], (-1, 1)))
X_cat.loc[:,'TicketPrefix'] =\
    ticket_prefix_imputer.transform(np.reshape(X_cat['TicketPrefix'], (-1, 1)))
X_t_cat.loc[:,'TicketPrefix'] =\
    ticket_prefix_imputer.transform(
        np.reshape(X_t_cat['TicketPrefix'], (-1, 1))
    )    
print('ticket_prefix_imputer values: ', ticket_prefix_imputer.statistics_,
      '\n\n')

# cat_mode_imputer
cat_mode_imputer = SimpleImputer(strategy='most_frequent')
cat_mode_imputer.fit(
    np.reshape(
        X_cat.loc[:,X_cat.columns.isin(['Sex', 'Embarked'])],\
        (-1, 2)
    )
)
X_cat.loc[:,X_cat.columns.isin(['Sex', 'Embarked'])] = \
    cat_mode_imputer.transform(np.reshape(
        X_cat.loc[:,X_cat.columns.isin(['Sex', 'Embarked'])],\
        (-1, 2))
    )
X_t_cat.loc[:,X_t_cat.columns.isin(['Sex', 'Embarked'])] = \
    cat_mode_imputer.transform(np.reshape(
        X_t_cat.loc[:,X_t_cat.columns.isin(['Sex', 'Embarked'])],\
        (-1, 2))
    )        
print('cat_mode_imputer values: ', cat_mode_imputer.statistics_, '\n\n')

# mean_imputer
mean_imputer = SimpleImputer(strategy='mean')
mean_imputer.fit(
    np.reshape(
        X_num.loc[:,X_num.columns.isin(['Age', 'Fare'])], (-1, 2)
    )
)
X_num.loc[:,X_num.columns.isin(['Age', 'Fare'])] = \
    mean_imputer.transform(np.reshape(
        X_num.loc[:,X_num.columns.isin(['Age', 'Fare'])], (-1, 2))
    )
X_t_num.loc[:,X_t_num.columns.isin(['Age', 'Fare'])] = \
    mean_imputer.transform(np.reshape(
        X_t_num.loc[:,X_t_num.columns.isin(['Age', 'Fare'])], (-1, 2))
    )    
print('mean_imputer values: ', mean_imputer.statistics_, '\n\n')

# num_mode_imputer
num_mode_imputer = SimpleImputer(strategy='most_frequent')
num_mode_imputer.fit(
    np.reshape(
        X_num.loc[:,X_num.columns.isin(['Pclass', 'SibSp', 'Parch'])],\
        (-1, 3)
    )
)
X_num.loc[:,X_num.columns.isin(['Pclass', 'SibSp', 'Parch'])] = \
    num_mode_imputer.transform(np.reshape(
        X_num.loc[:,X_num.columns.isin(['Pclass', 'SibSp', 'Parch'])],\
        (-1, 3))
    )
X_t_num.loc[:,X_t_num.columns.isin(['Pclass', 'SibSp', 'Parch'])] = \
    num_mode_imputer.transform(np.reshape(
        X_t_num.loc[:,X_t_num.columns.isin(['Pclass', 'SibSp', 'Parch'])],\
        (-1, 3))
    )
print('num_mode_imputer values: ', num_mode_imputer.statistics_, '\n\n')
```

    
    
    missing_imputer values:  ['missing_value' 'missing_value' 'missing_value'] 
    
    
    ticket_prefix_imputer values:  ['0'] 
    
    
    cat_mode_imputer values:  ['S' 'male'] 
    
    
    mean_imputer values:  [29.69911765 32.20420797] 
    
    
    num_mode_imputer values:  [3. 0. 0.] 
    
    
    

### Verify that there are no missing values after imputation


```python
# summary of missing values
print('Summary of Missing Values X:\n', X_cat.isnull().sum(), 
      '\n\n', X_num.isnull().sum(), '\n\n', sep='')
print('Summary of Missing Values X_t:\n', X_t_cat.isnull().sum(), 
      '\n\n', X_t_num.isnull().sum(), '\n\n', sep='')
```

    Summary of Missing Values X:
    Embarked        0
    Cabin           0
    Sex             0
    LastName        0
    TicketWord      0
    TicketPrefix    0
    dtype: int64
    
    Pclass    0
    SibSp     0
    Parch     0
    Age       0
    Fare      0
    dtype: int64
    
    
    Summary of Missing Values X_t:
    Embarked        0
    Cabin           0
    Sex             0
    LastName        0
    TicketWord      0
    TicketPrefix    0
    dtype: int64
    
    Pclass    0
    SibSp     0
    Parch     0
    Age       0
    Fare      0
    dtype: int64
    
    
    

### Encoding and Scaling

Categorical variables are hit with a one-hot encoder.  Numerical variables are scaled with `MinMaxScaler()`.


```python
# 1hot encoding and scaling
cat_encoder = OneHotEncoder(handle_unknown='ignore')
cat_encoder.fit(X_cat)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_num)

X_train = np.concatenate(
    (cat_encoder.transform(X_cat).toarray(), 
     min_max_scaler.transform(X_num)),
    axis=1
)
X_test = np.concatenate(
    (cat_encoder.transform(X_t_cat).toarray(), 
     min_max_scaler.transform(X_t_num)),
    axis=1
)
print('X_train.shape: ', X_train.shape)
print('X_test.shape: ', X_test.shape)
```

    X_train.shape:  (891, 1024)
    X_test.shape:  (418, 1024)
    

### Model Training

Logistic Regression, Support Vector Machine, and Random Forest models are trained.  I used Sci-Kit Learn's `GridSearchCV()` to tune the hyperparaments.  This method uses cross-validation to fit the model for all combinations of hyperparameters that are passed to it.  The model is then refit with the optimal hyperparameters.  The grids were updated over several attempts to refine them.


```python
# LogisticRegression
log_reg_param_grid = {
    'penalty': ['l2', 'l1'],
    'C': [0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10]
}
log_reg = LogisticRegression(solver='liblinear', max_iter=500)
log_reg_grid_search = GridSearchCV(
    log_reg,
    log_reg_param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)
log_reg_grid_search.fit(X_train, y_train)
log_reg_cvresults = log_reg_grid_search.cv_results_
for score, params in zip(log_reg_cvresults['mean_test_score'], 
                         log_reg_cvresults['params']):
    print(score, params)
print('Best parameters for log_reg: ', 
      log_reg_grid_search.best_params_, '\n\n')
```

    C:\anaconda3\lib\site-packages\sklearn\svm\_base.py:1242: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(
    

    0.8013181846713954 {'C': 0.1, 'penalty': 'l2'}
    0.7867365513778168 {'C': 0.1, 'penalty': 'l1'}
    0.8091833532107211 {'C': 0.2, 'penalty': 'l2'}
    0.7845081915761722 {'C': 0.2, 'penalty': 'l1'}
    0.817048521750047 {'C': 0.5, 'penalty': 'l2'}
    0.7856757265708367 {'C': 0.5, 'penalty': 'l1'}
    0.822666499278137 {'C': 1, 'penalty': 'l2'}
    0.810344611135522 {'C': 1, 'penalty': 'l1'}
    0.8316552633230808 {'C': 1.5, 'penalty': 'l2'}
    0.8181972255351203 {'C': 1.5, 'penalty': 'l1'}
    0.8339024543343168 {'C': 2, 'penalty': 'l2'}
    0.8226790534178645 {'C': 2, 'penalty': 'l1'}
    0.8361559224154165 {'C': 2.5, 'penalty': 'l2'}
    0.8305253907475991 {'C': 2.5, 'penalty': 'l1'}
    0.8406503044378884 {'C': 3, 'penalty': 'l2'}
    0.828278199736363 {'C': 3, 'penalty': 'l1'}
    0.8406628585776159 {'C': 3.5, 'penalty': 'l2'}
    0.8305316678174629 {'C': 3.5, 'penalty': 'l1'}
    0.8384282217061078 {'C': 4, 'penalty': 'l2'}
    0.8338899001945892 {'C': 4, 'penalty': 'l1'}
    0.8395580942815893 {'C': 4.5, 'penalty': 'l2'}
    0.8383842822170611 {'C': 4.5, 'penalty': 'l1'}
    0.8384344987759714 {'C': 5, 'penalty': 'l2'}
    0.8372606867114432 {'C': 5, 'penalty': 'l1'}
    0.8373109032703534 {'C': 5.5, 'penalty': 'l2'}
    0.8361370912058252 {'C': 5.5, 'penalty': 'l1'}
    0.8361873077647355 {'C': 6, 'penalty': 'l2'}
    0.8350134957002071 {'C': 6, 'penalty': 'l1'}
    0.8350637122591176 {'C': 6.5, 'penalty': 'l2'}
    0.8361370912058252 {'C': 6.5, 'penalty': 'l1'}
    0.8350637122591176 {'C': 7, 'penalty': 'l2'}
    0.8361308141359614 {'C': 7, 'penalty': 'l1'}
    0.8361873077647355 {'C': 10, 'penalty': 'l2'}
    0.8361496453455526 {'C': 10, 'penalty': 'l1'}
    Best parameters for log_reg:  {'C': 3.5, 'penalty': 'l2'} 
    
    
    


```python
# SupportVectorClassifier
svc_param_grid = [
    {
        'kernel': ['poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 5],
        'C': [0.1, 0.5, 1, 5, 10, 15, 20]
    },
    {
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto'],
        'C': [0.1, 0.5, 1, 5, 10, 15, 20]
    },
    {
        'kernel': ['sigmoid'],
        'gamma': ['scale', 'auto'],
        'C': [0.1, 0.5, 1, 5, 10, 15, 20]
    }
]
svc = SVC()
svc_grid_search = GridSearchCV(
    svc,
    svc_param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)
svc_grid_search.fit(X_train, y_train)
svc_cvresults = svc_grid_search.cv_results_
for score, params in zip(svc_cvresults['mean_test_score'], 
                         svc_cvresults['params']):
    print(score, params)
print('Best parameters for SVC: ', svc_grid_search.best_params_, '\n\n')
```

    0.7901198920343984 {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.7733098989391752 {'C': 0.1, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.1, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.1, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.800207143305505 {'C': 0.5, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.5, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8035904839620864 {'C': 0.5, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.5, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.7789341535371289 {'C': 0.5, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 0.5, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.8125729709371665 {'C': 1, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 1, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8237963718536188 {'C': 1, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 1, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.8170987383089574 {'C': 1, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 1, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.8440148138848785 {'C': 5, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 5, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8428974954491244 {'C': 5, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 5, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.8328227983177452 {'C': 5, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 5, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.8428974954491244 {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 10, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8451446864603603 {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 10, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.8328227983177454 {'C': 10, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 10, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.8428974954491244 {'C': 15, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 15, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8451384093904967 {'C': 15, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 15, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.8361747536250078 {'C': 15, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 15, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.8406503044378884 {'C': 20, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 20, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
    0.8451321323206328 {'C': 20, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 20, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
    0.8350511581193899 {'C': 20, 'degree': 5, 'gamma': 'scale', 'kernel': 'poly'}
    0.6161634548992531 {'C': 20, 'degree': 5, 'gamma': 'auto', 'kernel': 'poly'}
    0.7833783190006904 {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
    0.6161634548992531 {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'}
    0.7957064842131694 {'C': 0.5, 'gamma': 'scale', 'kernel': 'rbf'}
    0.6161634548992531 {'C': 0.5, 'gamma': 'auto', 'kernel': 'rbf'}
    0.8080848659845584 {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
    0.6161634548992531 {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
    0.8406503044378884 {'C': 5, 'gamma': 'scale', 'kernel': 'rbf'}
    0.7867365513778168 {'C': 5, 'gamma': 'auto', 'kernel': 'rbf'}
    0.8440148138848785 {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
    0.7867365513778168 {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
    0.8473793233318686 {'C': 15, 'gamma': 'scale', 'kernel': 'rbf'}
    0.7867365513778168 {'C': 15, 'gamma': 'auto', 'kernel': 'rbf'}
    0.8462557278262507 {'C': 20, 'gamma': 'scale', 'kernel': 'rbf'}
    0.7867365513778168 {'C': 20, 'gamma': 'auto', 'kernel': 'rbf'}
    0.7867365513778168 {'C': 0.1, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.6161634548992531 {'C': 0.1, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.7867365513778168 {'C': 0.5, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.6161634548992531 {'C': 0.5, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.795712761283033 {'C': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.6161634548992531 {'C': 1, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.7250831711756952 {'C': 5, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.7609880107965601 {'C': 5, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.7048898374238906 {'C': 10, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.7867365513778168 {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.7059883246500533 {'C': 15, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.7867365513778168 {'C': 15, 'gamma': 'auto', 'kernel': 'sigmoid'}
    0.7071119201556714 {'C': 20, 'gamma': 'scale', 'kernel': 'sigmoid'}
    0.7867365513778168 {'C': 20, 'gamma': 'auto', 'kernel': 'sigmoid'}
    Best parameters for SVC:  {'C': 15, 'gamma': 'scale', 'kernel': 'rbf'} 
    
    
    


```python
# RandomForestClassifier
rf_param_grid = {
    'n_estimators': [2, 4, 8, 16, 32, 64, 128],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [2, 4, 16, 32, 64],
    'max_features': ['sqrt', 'log2', None]
}
rf = RandomForestClassifier(n_jobs=-1)
rf_grid_search = GridSearchCV(
    rf,
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)
rf_grid_search.fit(X_train, y_train)
rf_cvresults = rf_grid_search.cv_results_
for score, params in zip(rf_cvresults['mean_test_score'], 
                         rf_cvresults['params']):
    print(score, params)
print('Best parameters for Random Forest: ', 
      rf_grid_search.best_params_, '\n\n')
```

    0.6622308706295901 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.6251522189441968 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.6184106459104891 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.6184106459104889 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.7115937480384156 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.675707739627142 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.6666813131630154 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.6666436507438329 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.6509446990145 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.6621869311405435 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.6487037850731279 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.7015002196974451 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.7419747661791476 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.7721863034335573 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.7980917707614086 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.7767120708053481 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.7958194714707176 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.7991525955683887 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.755407695687653 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.7789843700960392 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.7924172996045445 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.803665808800452 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.7912999811687904 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.8025233820852428 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.8159876969430669 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.7620990521624507 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.7968677421379701 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.8115058690603227 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.817079907099366 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.8159500345238841 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.8260561170045821 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.821568012051974 {'criterion': 'gini', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 128}
    0.6274056870252966 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.6375117695059946 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.6611009980541083 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.6217688782876153 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.6519615843324337 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.6184106459104889 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.617287050404871 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.6824555897307137 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.675707739627142 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.7273115309773398 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.7295210595693931 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.7025924298537441 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.7115937480384156 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.7216621680999309 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.760875023539012 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.7507877722679053 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.7587784822045069 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.7553637561986065 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.762111606302178 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.7677421379699956 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.7789467076768564 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.7272801456280209 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.7644152909421882 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.7531667817462809 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.7856757265708367 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.7946644906157806 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.8002887452137342 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.7935660033896177 {'criterion': 'gini', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 128}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.7867365513778168 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.7699516665620488 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.7867993220764548 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.7957818090515348 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.7901701085933087 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.7935346180402988 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.7867867679367271 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.7980227229929069 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.8181909484652564 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.7912811499591991 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.8215742891218379 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.8249325214989642 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.8137153976523759 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.8114744837110036 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.8193082669010107 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.8137153976523759 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.8226853304877283 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.8350260498399347 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.8327600276191074 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.8294143493817085 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.8238277572029377 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.8204506936162199 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.8058753373925052 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.8361433682756889 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.8305630531667818 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.8406691356474797 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.8384093904965162 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.8350386039796623 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.8384219446362439 {'criterion': 'gini', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 128}
    0.6476241290565564 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.6195279643462431 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.6891281149959199 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.6621555457912247 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.665570271797125 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.6441842947712008 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.6610570585650619 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.6599209089197163 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.6689473353838429 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.7610319502856067 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.7689285041742515 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.7666248195342413 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.7980101688531793 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.7845897934844015 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.7912937040989265 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.8002761910740066 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.7811499591990458 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.7676542589919026 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.7935911116690729 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.7834599209089198 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.8126043562864854 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.8148327160881299 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.8237900947837551 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.7452639507877723 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.7666562048835603 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.8014123407193521 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.8193145439708743 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.8047329106772958 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.824913690289373 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.8238026489234826 {'criterion': 'entropy', 'max_features': 'sqrt', 'max_leaf_nodes': 64, 'n_estimators': 128}
    0.6712196346745339 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.6229050279329609 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.6150461364634989 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.6620425585336764 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.6273805787458414 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.6206578369217249 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.617287050404871 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.6161634548992531 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.697012114744837 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.7161132383403428 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.6813382712949595 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.7059255539514154 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.7194777477873328 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.7216747222396585 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.7026049839934718 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.6880170736300295 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.7339966103822737 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.7643713514531416 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.7531353963969619 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.7699893289812316 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.7890653442972821 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.7688280710564308 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.7318059129998117 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.7631912623187497 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.7733224530789029 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.8125792480070304 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.8002887452137342 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.8002761910740066 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.8171175695185487 {'criterion': 'entropy', 'max_features': 'log2', 'max_leaf_nodes': 64, 'n_estimators': 128}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 2}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 4}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 8}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 16}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 32}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 64}
    0.7867365513778168 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 2, 'n_estimators': 128}
    0.7823112171238467 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 2}
    0.7800577490427468 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 4}
    0.80023225158496 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 8}
    0.7789341535371288 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 16}
    0.7823049400539828 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 32}
    0.7823049400539828 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 64}
    0.7924172996045448 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 4, 'n_estimators': 128}
    0.8148138848785387 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 2}
    0.8305191136777352 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 4}
    0.8137216747222398 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 8}
    0.8215491808423827 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 16}
    0.8204255853367648 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 32}
    0.8204130311970372 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 64}
    0.821542903772519 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 16, 'n_estimators': 128}
    0.8159876969430669 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 2}
    0.8294017952419811 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 4}
    0.8282907538760907 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 8}
    0.8327788588286987 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 16}
    0.8249325214989642 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 32}
    0.8316552633230808 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 64}
    0.8271671583704727 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 32, 'n_estimators': 128}
    0.8260561170045821 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 2}
    0.8215617349821104 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 4}
    0.8417927311530977 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 8}
    0.8339087314041805 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 16}
    0.8395329860021341 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 32}
    0.8406503044378884 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 64}
    0.8428912183792606 {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 128}
    Best parameters for Random Forest:  {'criterion': 'entropy', 'max_features': None, 'max_leaf_nodes': 64, 'n_estimators': 128} 
    
    
    

### Voting Classifier

`VotingClassifier()` uses a hard vote to make the final prediction for each sample in the test set.


```python
# VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg_grid_search.best_estimator_),
        ('svc', svc_grid_search.best_estimator_),
        ('rf', rf_grid_search.best_estimator_)
    ],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
voting_clf_predictions = voting_clf.predict(X_test)
solution = pd.DataFrame(
    data={'PassengerId': X_t['PassengerId'], 
          'Survived': voting_clf_predictions}
)
solution.to_csv('submission.csv', index=False)
```

### Conclusions

The accuracy might change from one run to the next because a random seed isn't used in training the models, but the version that ranked 227th had an accuracy of 0.80861, which felt unsatisfying to me.  I was especially surprised that using the voting ensemble didn't increase the accuracy much from what I was able to obtain with just a Logistic Regression model.  Investing more time in hyperparameter tuning might help, but I suspect that bigger gains could be realized with more rigorous feature engineering.  Rather than beat this horse to death, I intend to move on to other challenges on the Kaggle platform.  Thank you for reading!


```python

```
