These are the results of a Multivariate Analysis of Kentucky Derby data that spans the years 2002 through 2019. A variety of data about the horses that ran during those years were used in a binary logistic regression to predict the probability of winning. The model was fit with data from 2002-2018 and used to predict the probability of winning for the 2019 field. For 2002-2018, the model successfully ranked the winner 9/17 times, though the cross-validation by year was not rigorous in the sense that the model was not refit by successively withholding the fields by year. The 145th running of the Kentucky Derby was on May 4, 2019.


```python
import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder()
```


```python
# get data
path = r'C:\Users\Zen Warrior\OneDrive\Kentucky Derby'
filename = r'derbyData.2022.xlsx'
pastPerformance_import = pd.read_excel((path + '\\' + filename), sheet_name='derbyData')
pastPerformance = pastPerformance_import.copy()
pastPerformance.set_index('name', inplace=True)

field_import = pd.read_excel((path + '\\' + filename), sheet_name='currentField')
field = field_import.copy()
field.set_index('name', inplace=True)
field.drop('finish', axis=1, inplace=True)
```


```python
# summarize data frame
print('Past Performance Data:\n', pastPerformance, '\n\n', sep='')
print('Past Performance Data Types:\n', pastPerformance.dtypes, '\n\n', sep='')
```

    Past Performance Data:
                      year  finish  odds style  post  races  win  place  show  \
    name                                                                        
    KnownAgenda       2021       8   9.9     P     1      6    3      1     1   
    LikeTheKing       2021      11  49.9    EP     2      6    3      2     1   
    BrooklynStrong    2021      14  43.5    EP     3      5    3      0     1   
    KeepMeInMind      2021       6  49.0     S     4      6    1      2     1   
    Sainthood         2021      10  43.4    EP     5      3    1      2     0   
    ...                ...     ...   ...   ...   ...    ...  ...    ...   ...   
    CameHome          2002       6   8.2    EP    14      7    6      0     0   
    Saarland          2002      10   6.9     S    15      7    2      1     0   
    Itsallinthechase  2002      16  94.5     S    16     14    2      3     2   
    EasyGrades        2002      13  43.8     P    17      5    1      3     0   
    BlueBurner        2002      11  24.2     S    18      6    3      1     1   
    
                      streak  ...     e1     e2   late  rcgspdavg   bspd  avgcl3  \
    name                      ...                                                  
    KnownAgenda            2  ...   83.0   93.0  112.0       92.0  101.0   118.8   
    LikeTheKing            1  ...   98.0  102.0   96.0       91.0   94.0   116.9   
    BrooklynStrong         0  ...   84.0   90.0   93.0       95.0   88.0   117.9   
    KeepMeInMind           0  ...   86.0   94.0  100.0       87.0   84.0   116.4   
    Sainthood              0  ...   94.0   93.0   90.0       91.0   93.0   117.2   
    ...                  ...  ...    ...    ...    ...        ...    ...     ...   
    CameHome               3  ...   91.0   99.0  103.0      100.0   95.0   121.8   
    Saarland               0  ...  103.0  114.0  110.0      100.0  104.0   120.9   
    Itsallinthechase       0  ...   89.0   92.0  108.0       89.0   86.0   118.3   
    EasyGrades             0  ...   88.0   96.0  106.0       92.0   92.0   118.9   
    BlueBurner             0  ...   89.0  110.0  100.0      102.0  106.0   121.6   
    
                         pp  cd  class  stam  
    name                                      
    KnownAgenda       142.0 NaN    NaN   NaN  
    LikeTheKing       139.0 NaN    NaN   NaN  
    BrooklynStrong    138.1 NaN    NaN   NaN  
    KeepMeInMind      138.4 NaN    NaN   NaN  
    Sainthood         136.1 NaN    NaN   NaN  
    ...                 ...  ..    ...   ...  
    CameHome          149.6 NaN    NaN   NaN  
    Saarland          147.8 NaN    NaN   NaN  
    Itsallinthechase  134.6 NaN    NaN   NaN  
    EasyGrades        141.2 NaN    NaN   NaN  
    BlueBurner        152.6 NaN    NaN   NaN  
    
    [377 rows x 25 columns]
    
    
    Past Performance Data Types:
    year           int64
    finish         int64
    odds         float64
    style         object
    post           int64
    races          int64
    win            int64
    place          int64
    show           int64
    streak         int64
    finlr          int64
    gmoney         int64
    gwins          int64
    g1money        int64
    g1wins         int64
    e1           float64
    e2           float64
    late         float64
    rcgspdavg    float64
    bspd         float64
    avgcl3       float64
    pp           float64
    cd           float64
    class        float64
    stam         float64
    dtype: object
    
    
    


```python
# drop horses with finish == 0
print('Horses with No Finish Position:\n', pastPerformance.loc[pastPerformance.finish == 0], '\n')
print('Number of Horses to Drop:', len(pastPerformance.loc[pastPerformance.finish == 0]))
pastPerformance.drop(index=pastPerformance.loc[pastPerformance.finish == 0].index, inplace=True)
print('New Size of pastPerformance: ', pastPerformance.shape, '\n\n')

# create target variable and drop 'finish'
y_train = np.array((pastPerformance['finish'] == 1).astype(np.int32))
pastPerformance.drop('finish', axis=1, inplace=True)

# summary of missing values
print('Summary of Missing Values:\n', pastPerformance.isnull().sum(), '\n\n', sep='')
print('cd is only available from 2003-2015\n', 'class and stam are only available from 2010-2015\n',
      'These variables will be dropped', sep='')
pastPerformance.drop(columns=['cd', 'class', 'stam'], inplace=True)
print('New Size of pastPerformance: ', pastPerformance.shape, '\n\n')
```

    Horses with No Finish Position:
                   year  finish  odds style  post  races  win  place  show  streak  \
    name                                                                            
    MedinaSpirit  2021       0  12.0     E     8      5    2      3     0       0   
    
                  ...     e1     e2  late  rcgspdavg  bspd  avgcl3     pp  cd  \
    name          ...                                                           
    MedinaSpirit  ...  106.0  106.0  94.0       95.0  98.0   119.5  148.9 NaN   
    
                  class  stam  
    name                       
    MedinaSpirit    NaN   NaN  
    
    [1 rows x 25 columns] 
    
    Number of Horses to Drop: 1
    New Size of pastPerformance:  (376, 25) 
    
    
    Summary of Missing Values:
    year           0
    odds           1
    style          5
    post           0
    races          0
    win            0
    place          0
    show           0
    streak         0
    finlr          0
    gmoney         0
    gwins          0
    g1money        0
    g1wins         0
    e1            10
    e2            10
    late          10
    rcgspdavg      6
    bspd          37
    avgcl3         4
    pp             9
    cd           128
    class        263
    stam         263
    dtype: int64
    
    
    cd is only available from 2003-2015
    class and stam are only available from 2010-2015
    These variables will be dropped
    New Size of pastPerformance:  (376, 21) 
    
    
    


```python
for col in pastPerformance:
    
    if col not in ['name']:
        
        print('Series Name: ', col)
        print('Series Type: ', pastPerformance[col].dtype)
        
        # if pastPerformance[col].dtype == 'object':
        #     print('Unique Values:\n', pd.unique(pastPerformance[col]), sep='')
        # else:
        #     print('Unique Values:\n', np.sort(pd.unique(pastPerformance[col])), sep='')
            
        print('Number of Missing Values: ', pastPerformance[col].isnull().sum())

        if (pastPerformance[col].dtype == 'int64') & (col not in ['year', 'finish', 'post']):
            colMode = mode(pastPerformance[col])
            print(col, 'Mode: ', colMode)
            if pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(pastPerformance[col]))
                pastPerformance.loc[np.isnan(pastPerformance[col]), [col]] = colMode
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col], '\n'])

        elif pastPerformance[col].dtype == 'float64':
            colMean = np.nanmean(pastPerformance[col])
            print('Mean ', col, ':  ', round(colMean, 1), sep='')
            if pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[np.isnan(pastPerformance[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(pastPerformance[col]))
                pastPerformance.loc[np.isnan(pastPerformance[col]), [col]] = colMean
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col]], '\n')

        elif pastPerformance[col].dtype == 'object':
            colMode = mode(pastPerformance[col])
            print(col, 'Mode: ', colMode)
            if pastPerformance.loc[pastPerformance[col].isna(), ['year', col]].empty == False:
                print('Horses to Impute:\n', pastPerformance.loc[pastPerformance[col].isna(), ['year', col]], '\n')
                imputedIndices = pd.Index(pastPerformance[col].isna())
                pastPerformance.loc[pastPerformance[col].isna(), [col]] = colMode
                print('After Imputation:\n', pastPerformance.loc[imputedIndices, ['year', col]], '\n')

        print('\n\n')
```

    Series Name:  year
    Series Type:  int64
    Number of Missing Values:  0
    
    
    
    Series Name:  odds
    Series Type:  float64
    Number of Missing Values:  1
    Mean odds:  26.6
    Horses to Impute:
                 year  odds
    name                  
    WildHorses  2002   NaN 
    
    After Imputation:
                 year       odds
    name                       
    WildHorses  2002  26.571733 
    
    
    
    
    Series Name:  style
    Series Type:  object
    Number of Missing Values:  5
    style Mode:  EP
    Horses to Impute:
                     year style
    name                      
    MasterFencer    2019   NaN
    Lani            2016   NaN
    Mubtaahij       2015   NaN
    DaddyLongLegs   2012   NaN
    CastleGandolfo  2002   NaN 
    
    After Imputation:
                     year style
    name                      
    MasterFencer    2019    EP
    Lani            2016    EP
    Mubtaahij       2015    EP
    DaddyLongLegs   2012    EP
    CastleGandolfo  2002    EP 
    
    
    
    
    Series Name:  post
    Series Type:  int64
    Number of Missing Values:  0
    
    
    
    Series Name:  races
    Series Type:  int64
    Number of Missing Values:  0
    races Mode:  6
    
    
    
    Series Name:  win
    Series Type:  int64
    Number of Missing Values:  0
    win Mode:  3
    
    
    
    Series Name:  place
    Series Type:  int64
    Number of Missing Values:  0
    place Mode:  1
    
    
    
    Series Name:  show
    Series Type:  int64
    Number of Missing Values:  0
    show Mode:  0
    
    
    
    Series Name:  streak
    Series Type:  int64
    Number of Missing Values:  0
    streak Mode:  0
    
    
    
    Series Name:  finlr
    Series Type:  int64
    Number of Missing Values:  0
    finlr Mode:  1
    
    
    
    Series Name:  gmoney
    Series Type:  int64
    Number of Missing Values:  0
    gmoney Mode:  2
    
    
    
    Series Name:  gwins
    Series Type:  int64
    Number of Missing Values:  0
    gwins Mode:  1
    
    
    
    Series Name:  g1money
    Series Type:  int64
    Number of Missing Values:  0
    g1money Mode:  0
    
    
    
    Series Name:  g1wins
    Series Type:  int64
    Number of Missing Values:  0
    g1wins Mode:  0
    
    
    
    Series Name:  e1
    Series Type:  float64
    Number of Missing Values:  10
    Mean e1:  94.0
    Horses to Impute:
                     year  e1
    name                    
    MasterFencer    2019 NaN
    Mendelssohn     2018 NaN
    Lani            2016 NaN
    Mubtaahij       2015 NaN
    LinesOfBattle   2013 NaN
    Trinniberg      2012 NaN
    MasterOfHounds  2011 NaN
    DesertParty     2009 NaN
    WildHorses      2002 NaN
    CastleGandolfo  2002 NaN 
    
    After Imputation:
                     year         e1
    name                           
    MasterFencer    2019  94.038251
    Mendelssohn     2018  94.038251
    Lani            2016  94.038251
    Mubtaahij       2015  94.038251
    LinesOfBattle   2013  94.038251
    Trinniberg      2012  94.038251
    MasterOfHounds  2011  94.038251
    DesertParty     2009  94.038251
    WildHorses      2002  94.038251
    CastleGandolfo  2002  94.038251 
    
    
    
    
    Series Name:  e2
    Series Type:  float64
    Number of Missing Values:  10
    Mean e2:  102.4
    Horses to Impute:
                     year  e2
    name                    
    MasterFencer    2019 NaN
    Mendelssohn     2018 NaN
    Lani            2016 NaN
    Mubtaahij       2015 NaN
    LinesOfBattle   2013 NaN
    Trinniberg      2012 NaN
    MasterOfHounds  2011 NaN
    DesertParty     2009 NaN
    WildHorses      2002 NaN
    CastleGandolfo  2002 NaN 
    
    After Imputation:
                     year         e2
    name                           
    MasterFencer    2019  102.36612
    Mendelssohn     2018  102.36612
    Lani            2016  102.36612
    Mubtaahij       2015  102.36612
    LinesOfBattle   2013  102.36612
    Trinniberg      2012  102.36612
    MasterOfHounds  2011  102.36612
    DesertParty     2009  102.36612
    WildHorses      2002  102.36612
    CastleGandolfo  2002  102.36612 
    
    
    
    
    Series Name:  late
    Series Type:  float64
    Number of Missing Values:  10
    Mean late:  103.4
    Horses to Impute:
                     year  late
    name                      
    MasterFencer    2019   NaN
    Mendelssohn     2018   NaN
    Lani            2016   NaN
    Mubtaahij       2015   NaN
    LinesOfBattle   2013   NaN
    Trinniberg      2012   NaN
    MasterOfHounds  2011   NaN
    DesertParty     2009   NaN
    WildHorses      2002   NaN
    CastleGandolfo  2002   NaN 
    
    After Imputation:
                     year        late
    name                            
    MasterFencer    2019  103.385246
    Mendelssohn     2018  103.385246
    Lani            2016  103.385246
    Mubtaahij       2015  103.385246
    LinesOfBattle   2013  103.385246
    Trinniberg      2012  103.385246
    MasterOfHounds  2011  103.385246
    DesertParty     2009  103.385246
    WildHorses      2002  103.385246
    CastleGandolfo  2002  103.385246 
    
    
    
    
    Series Name:  rcgspdavg
    Series Type:  float64
    Number of Missing Values:  6
    Mean rcgspdavg:  95.7
    Horses to Impute:
                     year  rcgspdavg
    name                           
    MasterFencer    2019        NaN
    Lani            2016        NaN
    Mubtaahij       2015        NaN
    OuttaHere       2003        NaN
    WildHorses      2002        NaN
    EssenceOfDubai  2002        NaN 
    
    After Imputation:
                     year  rcgspdavg
    name                           
    MasterFencer    2019  95.705405
    Lani            2016  95.705405
    Mubtaahij       2015  95.705405
    OuttaHere       2003  95.705405
    WildHorses      2002  95.705405
    EssenceOfDubai  2002  95.705405 
    
    
    
    
    Series Name:  bspd
    Series Type:  float64
    Number of Missing Values:  37
    Mean bspd:  98.2
    Horses to Impute:
                     year  bspd
    name                      
    Helium          2021   NaN
    GrayMagician    2019   NaN
    Improbable      2019   NaN
    PlusQueParfait  2019   NaN
    MasterFencer    2019   NaN
    LongRangeToddy  2019   NaN
    Mendelssohn     2018   NaN
    TrojanNation    2016   NaN
    Lani            2016   NaN
    Destin          2016   NaN
    Exaggerator     2016   NaN
    Outwork         2016   NaN
    MorSpirit       2016   NaN
    DanzingCandy    2016   NaN
    Mubtaahij       2015   NaN
    LinesOfBattle   2013   NaN
    WillTakeCharge  2013   NaN
    DaddyLongLegs   2012   NaN
    Trinniberg      2012   NaN
    DerbyKitten     2011   NaN
    MasterOfHounds  2011   NaN
    FriesanFire     2009   NaN
    RegalRansom     2009   NaN
    DesertParty     2009   NaN
    CircularQuay    2007   NaN
    KeyedEntry      2006   NaN
    ShowingUp       2006   NaN
    DeputyGlitters  2006   NaN
    MinisterEric    2004   NaN
    ProPrado        2004   NaN
    EyeOfTheTiger   2003   NaN
    OuttaHere       2003   NaN
    Scrimshaw       2003   NaN
    Johannesburg    2002   NaN
    WildHorses      2002   NaN
    EssenceOfDubai  2002   NaN
    CastleGandolfo  2002   NaN 
    
    After Imputation:
                     year      bspd
    name                          
    Helium          2021  98.19174
    GrayMagician    2019  98.19174
    Improbable      2019  98.19174
    PlusQueParfait  2019  98.19174
    MasterFencer    2019  98.19174
    LongRangeToddy  2019  98.19174
    Mendelssohn     2018  98.19174
    TrojanNation    2016  98.19174
    Lani            2016  98.19174
    Destin          2016  98.19174
    Exaggerator     2016  98.19174
    Outwork         2016  98.19174
    MorSpirit       2016  98.19174
    DanzingCandy    2016  98.19174
    Mubtaahij       2015  98.19174
    LinesOfBattle   2013  98.19174
    WillTakeCharge  2013  98.19174
    DaddyLongLegs   2012  98.19174
    Trinniberg      2012  98.19174
    DerbyKitten     2011  98.19174
    MasterOfHounds  2011  98.19174
    FriesanFire     2009  98.19174
    RegalRansom     2009  98.19174
    DesertParty     2009  98.19174
    CircularQuay    2007  98.19174
    KeyedEntry      2006  98.19174
    ShowingUp       2006  98.19174
    DeputyGlitters  2006  98.19174
    MinisterEric    2004  98.19174
    ProPrado        2004  98.19174
    EyeOfTheTiger   2003  98.19174
    OuttaHere       2003  98.19174
    Scrimshaw       2003  98.19174
    Johannesburg    2002  98.19174
    WildHorses      2002  98.19174
    EssenceOfDubai  2002  98.19174
    CastleGandolfo  2002  98.19174 
    
    
    
    
    Series Name:  avgcl3
    Series Type:  float64
    Number of Missing Values:  4
    Mean avgcl3:  119.3
    Horses to Impute:
                   year  avgcl3
    name                      
    MasterFencer  2019     NaN
    Lani          2016     NaN
    Mubtaahij     2015     NaN
    WildHorses    2002     NaN 
    
    After Imputation:
                   year      avgcl3
    name                          
    MasterFencer  2019  119.305645
    Lani          2016  119.305645
    Mubtaahij     2015  119.305645
    WildHorses    2002  119.305645 
    
    
    
    
    Series Name:  pp
    Series Type:  float64
    Number of Missing Values:  9
    Mean pp:  144.0
    Horses to Impute:
                     year  pp
    name                    
    MasterFencer    2019 NaN
    Mendelssohn     2018 NaN
    Lani            2016 NaN
    Mubtaahij       2015 NaN
    RegalRansom     2009 NaN
    DesertParty     2009 NaN
    WildHorses      2002 NaN
    EssenceOfDubai  2002 NaN
    CastleGandolfo  2002 NaN 
    
    After Imputation:
                     year          pp
    name                            
    MasterFencer    2019  143.975204
    Mendelssohn     2018  143.975204
    Lani            2016  143.975204
    Mubtaahij       2015  143.975204
    RegalRansom     2009  143.975204
    DesertParty     2009  143.975204
    WildHorses      2002  143.975204
    EssenceOfDubai  2002  143.975204
    CastleGandolfo  2002  143.975204 
    
    
    
    
    


```python
# scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train_num = standard_scaler.fit_transform(np.array(pastPerformance.drop(['year', 'style'], axis=1)))
```


```python
# one-hot encoding
print('One-Hot Style:\n', one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray())
print('X_train_num:\n', X_train_num)
X_train = np.concatenate((X_train_num,
                          one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray()),
                          axis=1)
print('X_train Type:\n', type(X_train))
print('X_train Shape:\n', X_train.shape)
print('One-Hot Style Shape:\n', one_hot_encoder.fit_transform(pastPerformance[['style']]).toarray().shape)
print('X_train_num Shape:\n', X_train_num.shape)
```

    One-Hot Style:
     [[0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     ...
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    X_train_num:
     [[-0.93837595 -1.63138209 -0.33780363 ...  0.42132182 -0.27760246
      -0.34625531]
     [ 1.3130419  -1.45526964 -0.33780363 ... -0.62888478 -1.32071472
      -0.87215832]
     [ 0.95281505 -1.27915719 -0.80649058 ... -1.52906186 -0.77170826
      -1.02992922]
     ...
     [ 3.82337281  1.01030467  3.41169199 ... -1.82912089 -0.55210568
      -1.64348274]
     [ 0.96970068  1.18641712 -0.80649058 ... -0.92894381 -0.22270181
      -0.48649611]
     [-0.13349407  1.36252957 -0.33780363 ...  1.17146939  1.25961561
       1.51193534]]
    X_train Type:
     <class 'numpy.ndarray'>
    X_train Shape:
     (376, 23)
    One-Hot Style Shape:
     (376, 4)
    X_train_num Shape:
     (376, 19)
    


```python
# summarize data frame
print('Current Field Data:\n', field, '\n\n', sep='')
print('Current Field Data Types:\n', field.dtypes, '\n\n', sep='')
```

    Current Field Data:
                      year  odds style  post  races  win  place  show  streak  \
    name                                                                        
    MoDonegal         2022     9     S     1      5    3      0     2       1   
    Happy Jack        2022    20     S     2      4    1      0     2       0   
    Epicenter         2022     5    EP     3      6    4      1     0       2   
    SummerIsTomorrow  2022    33   NaN     4      7    2      3     0       0   
    SmileHappy        2022    15     P     5      4    2      2     0       0   
    Messier           2022     6    EP     6      6    3      3     0       0   
    CrownPride        2022    17   NaN     7      4    3      0     0       1   
    ChargeIt          2022    14    EP     8      3    1      2     0       0   
    TizTheBomb        2022    26     P     9      8    5      1     0       2   
    Zandon            2022     7     S    10      4    2      1     1       1   
    PioneerOfMedina   2022    52    EP    11      6    2      1     2       0   
    Taiba             2022     5    EP    12      2    2      0     0       2   
    Simplification    2022    41    EP    13      7    3      1     2       0   
    BarberRoad        2022    50     S    14      8    2      4     1       0   
    WhiteAbarrio      2022    13    EP    15      5    4      0     1       2   
    Cyberknife        2022    12    EP    16      6    3      2     0       2   
    ClassicCauseway   2022    76    EP    17      6    3      1     1       0   
    TawnyPort         2022    77     P    18      5    3      1     0       1   
    Zozos             2022    45    EP    19      3    2      1     0       0   
    RichStrike        2022    99     S    21      7    1      0     3       0   
    
                      finlr  ...     e1     e2   late  rcgspdavg   bspd  avgcl3  \
    name                     ...                                                  
    MoDonegal             1  ...   91.0   95.0  121.0       99.0  111.0   120.2   
    Happy Jack            3  ...   90.0   98.0   95.0       86.0   97.0   114.9   
    Epicenter             1  ...  104.0  103.0  106.0       97.0  101.0   120.1   
    SummerIsTomorrow      2  ...    NaN    NaN    NaN        NaN    NaN     NaN   
    SmileHappy            2  ...   92.0   96.0  102.0       98.0  101.0   119.7   
    Messier               2  ...  100.0  109.0  102.0      104.0  108.0   120.1   
    CrownPride            1  ...    NaN    NaN    NaN        NaN    NaN     NaN   
    ChargeIt              2  ...   96.0  108.0  107.0       94.0   94.0   118.8   
    TizTheBomb            1  ...   87.0   93.0  105.0       88.0  101.0   117.0   
    Zandon                1  ...   89.0   92.0  116.0       97.0  103.0   120.0   
    PioneerOfMedina       3  ...   98.0  100.0   95.0       92.0   96.0   117.9   
    Taiba                 1  ...   96.0  106.0  109.0      106.0  111.0   119.4   
    Simplification        3  ...  100.0  113.0   95.0       93.0   93.0   119.8   
    BarberRoad            2  ...   90.0   96.0  108.0       91.0   91.0   118.6   
    WhiteAbarrio          1  ...   98.0  110.0  112.0       95.0   96.0   119.7   
    Cyberknife            1  ...   97.0   99.0   91.0       90.0   94.0   117.9   
    ClassicCauseway      11  ...  103.0  107.0   98.0       87.0   72.0   117.6   
    TawnyPort             1  ...   91.0   94.0   99.0       93.0   99.0   118.5   
    Zozos                 2  ...   90.0   93.0  111.0       89.0   98.0   118.1   
    RichStrike            3  ...   90.0   88.0  107.0       90.0   95.0   115.9   
    
                         pp  cd  class  stam  
    name                                      
    MoDonegal         147.5 NaN    NaN   NaN  
    Happy Jack        136.5 NaN    NaN   NaN  
    Epicenter         146.5 NaN    NaN   NaN  
    SummerIsTomorrow    NaN NaN    NaN   NaN  
    SmileHappy        146.9 NaN    NaN   NaN  
    Messier           148.9 NaN    NaN   NaN  
    CrownPride          NaN NaN    NaN   NaN  
    ChargeIt          146.7 NaN    NaN   NaN  
    TizTheBomb        146.1 NaN    NaN   NaN  
    Zandon            147.1 NaN    NaN   NaN  
    PioneerOfMedina   139.2 NaN    NaN   NaN  
    Taiba             144.5 NaN    NaN   NaN  
    Simplification    148.8 NaN    NaN   NaN  
    BarberRoad        138.0 NaN    NaN   NaN  
    WhiteAbarrio      148.1 NaN    NaN   NaN  
    Cyberknife        142.1 NaN    NaN   NaN  
    ClassicCauseway   143.3 NaN    NaN   NaN  
    TawnyPort         142.8 NaN    NaN   NaN  
    Zozos             139.5 NaN    NaN   NaN  
    RichStrike        132.0 NaN    NaN   NaN  
    
    [20 rows x 24 columns]
    
    
    Current Field Data Types:
    year           int64
    odds           int64
    style         object
    post           int64
    races          int64
    win            int64
    place          int64
    show           int64
    streak         int64
    finlr          int64
    gmoney         int64
    gwins          int64
    g1money        int64
    g1wins         int64
    e1           float64
    e2           float64
    late         float64
    rcgspdavg    float64
    bspd         float64
    avgcl3       float64
    pp           float64
    cd           float64
    class        float64
    stam         float64
    dtype: object
    
    
    


```python
# summary of missing values
print('Summary of Missing Values:\n', field.isnull().sum(), '\n\n', sep='')
print('cd is only available from 2003-2015\n', 'class and stam are only available from 2010-2015\n',
      'These variables will be dropped', sep='')
field.drop(columns=['cd', 'class', 'stam'], inplace=True)
print('Size of field: ', field.shape, '\n\n')
```

    Summary of Missing Values:
    year          0
    odds          0
    style         2
    post          0
    races         0
    win           0
    place         0
    show          0
    streak        0
    finlr         0
    gmoney        0
    gwins         0
    g1money       0
    g1wins        0
    e1            2
    e2            2
    late          2
    rcgspdavg     2
    bspd          2
    avgcl3        2
    pp            2
    cd           20
    class        20
    stam         20
    dtype: int64
    
    
    cd is only available from 2003-2015
    class and stam are only available from 2010-2015
    These variables will be dropped
    Size of field:  (20, 21) 
    
    
    


```python
for col in field:
    
    if col not in ['name']:
        
        print('Series Name: ', col)
        print('Series Type: ', field[col].dtype)
        
        # if field[col].dtype == 'object':
        #     print('Unique Values:\n', pd.unique(field[col]), sep='')
        # else:
        #     print('Unique Values:\n', np.sort(pd.unique(field[col])), sep='')
            
        print('Number of Missing Values: ', field[col].isnull().sum())

        if (field[col].dtype == 'int64') & (col not in ['year', 'finish', 'post']):
            colMode = mode(field[col])
            print(col, 'Mode: ', colMode)
            if field.loc[np.isnan(field[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[np.isnan(field[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(field[col]))
                field.loc[np.isnan(field[col]), [col]] = colMode
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col], '\n'])

        elif field[col].dtype == 'float64':
            colMean = np.nanmean(field[col])
            print('Mean ', col, ':  ', round(colMean, 1), sep='')
            if field.loc[np.isnan(field[col]), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[np.isnan(field[col]), ['year', col]], '\n')
                imputedIndices = pd.Index(np.isnan(field[col]))
                field.loc[np.isnan(field[col]), [col]] = colMean
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col]], '\n')

        elif field[col].dtype == 'object':
            colMode = mode(field[col])
            print(col, 'Mode: ', colMode)
            if field.loc[field[col].isna(), ['year', col]].empty == False:
                print('Horses to Impute:\n', field.loc[field[col].isna(), ['year', col]], '\n')
                imputedIndices = pd.Index(field[col].isna())
                field.loc[field[col].isna(), [col]] = colMode
                print('After Imputation:\n', field.loc[imputedIndices, ['year', col]], '\n')

        print('\n\n')
```

    Series Name:  year
    Series Type:  int64
    Number of Missing Values:  0
    
    
    
    Series Name:  odds
    Series Type:  int64
    Number of Missing Values:  0
    odds Mode:  5
    
    
    
    Series Name:  style
    Series Type:  object
    Number of Missing Values:  2
    style Mode:  EP
    Horses to Impute:
                       year style
    name                        
    SummerIsTomorrow  2022   NaN
    CrownPride        2022   NaN 
    
    After Imputation:
                       year style
    name                        
    SummerIsTomorrow  2022    EP
    CrownPride        2022    EP 
    
    
    
    
    Series Name:  post
    Series Type:  int64
    Number of Missing Values:  0
    
    
    
    Series Name:  races
    Series Type:  int64
    Number of Missing Values:  0
    races Mode:  6
    
    
    
    Series Name:  win
    Series Type:  int64
    Number of Missing Values:  0
    win Mode:  3
    
    
    
    Series Name:  place
    Series Type:  int64
    Number of Missing Values:  0
    place Mode:  1
    
    
    
    Series Name:  show
    Series Type:  int64
    Number of Missing Values:  0
    show Mode:  0
    
    
    
    Series Name:  streak
    Series Type:  int64
    Number of Missing Values:  0
    streak Mode:  0
    
    
    
    Series Name:  finlr
    Series Type:  int64
    Number of Missing Values:  0
    finlr Mode:  1
    
    
    
    Series Name:  gmoney
    Series Type:  int64
    Number of Missing Values:  0
    gmoney Mode:  1
    
    
    
    Series Name:  gwins
    Series Type:  int64
    Number of Missing Values:  0
    gwins Mode:  0
    
    
    
    Series Name:  g1money
    Series Type:  int64
    Number of Missing Values:  0
    g1money Mode:  1
    
    
    
    Series Name:  g1wins
    Series Type:  int64
    Number of Missing Values:  0
    g1wins Mode:  0
    
    
    
    Series Name:  e1
    Series Type:  float64
    Number of Missing Values:  2
    Mean e1:  94.6
    Horses to Impute:
                       year  e1
    name                      
    SummerIsTomorrow  2022 NaN
    CrownPride        2022 NaN 
    
    After Imputation:
                       year         e1
    name                             
    SummerIsTomorrow  2022  94.555556
    CrownPride        2022  94.555556 
    
    
    
    
    Series Name:  e2
    Series Type:  float64
    Number of Missing Values:  2
    Mean e2:  100.0
    Horses to Impute:
                       year  e2
    name                      
    SummerIsTomorrow  2022 NaN
    CrownPride        2022 NaN 
    
    After Imputation:
                       year     e2
    name                         
    SummerIsTomorrow  2022  100.0
    CrownPride        2022  100.0 
    
    
    
    
    Series Name:  late
    Series Type:  float64
    Number of Missing Values:  2
    Mean late:  104.4
    Horses to Impute:
                       year  late
    name                        
    SummerIsTomorrow  2022   NaN
    CrownPride        2022   NaN 
    
    After Imputation:
                       year        late
    name                              
    SummerIsTomorrow  2022  104.388889
    CrownPride        2022  104.388889 
    
    
    
    
    Series Name:  rcgspdavg
    Series Type:  float64
    Number of Missing Values:  2
    Mean rcgspdavg:  93.8
    Horses to Impute:
                       year  rcgspdavg
    name                             
    SummerIsTomorrow  2022        NaN
    CrownPride        2022        NaN 
    
    After Imputation:
                       year  rcgspdavg
    name                             
    SummerIsTomorrow  2022  93.833333
    CrownPride        2022  93.833333 
    
    
    
    
    Series Name:  bspd
    Series Type:  float64
    Number of Missing Values:  2
    Mean bspd:  97.8
    Horses to Impute:
                       year  bspd
    name                        
    SummerIsTomorrow  2022   NaN
    CrownPride        2022   NaN 
    
    After Imputation:
                       year       bspd
    name                             
    SummerIsTomorrow  2022  97.833333
    CrownPride        2022  97.833333 
    
    
    
    
    Series Name:  avgcl3
    Series Type:  float64
    Number of Missing Values:  2
    Mean avgcl3:  118.6
    Horses to Impute:
                       year  avgcl3
    name                          
    SummerIsTomorrow  2022     NaN
    CrownPride        2022     NaN 
    
    After Imputation:
                       year      avgcl3
    name                              
    SummerIsTomorrow  2022  118.566667
    CrownPride        2022  118.566667 
    
    
    
    
    Series Name:  pp
    Series Type:  float64
    Number of Missing Values:  2
    Mean pp:  143.6
    Horses to Impute:
                       year  pp
    name                      
    SummerIsTomorrow  2022 NaN
    CrownPride        2022 NaN 
    
    After Imputation:
                       year          pp
    name                              
    SummerIsTomorrow  2022  143.583333
    CrownPride        2022  143.583333 
    
    
    
    
    


```python
# scaling
X_test_num = standard_scaler.transform(np.array(field.drop(['year', 'style'], axis=1)))
```


```python
# one-hot encoding
print('One-Hot Style:\n', one_hot_encoder.transform(field[['style']]).toarray())
print('X_test_num:\n', X_test_num)
X_test = np.concatenate((X_test_num,
                          one_hot_encoder.transform(field[['style']]).toarray()),
                          axis=1)
print('X_train Type:\n', type(X_test))
print('X_train Shape:\n', X_test.shape)
print('One-Hot Style Shape:\n', one_hot_encoder.transform(field[['style']]).toarray().shape)
print('X_test_num Shape:\n', X_test_num.shape)
```

    One-Hot Style:
     [[0. 0. 0. 1.]
     [0. 0. 0. 1.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]
    X_test_num:
     [[-0.98903285 -1.63138209 -0.80649058  0.15634265 -1.26542223  1.11774086
       0.21348179 -0.75339317  0.4502701   0.77349075 -0.86904655 -0.53395584
      -0.3621822  -0.83778594  2.26809697  0.63042608  1.92161696  0.49100658
       0.61790022]
     [-0.36989294 -1.45526964 -1.27517753 -1.41125301 -1.26542223  1.11774086
      -0.60559119  0.2200633  -0.29555174 -1.04991064  0.18502281 -0.53395584
      -0.48138965 -0.49658084 -1.07969437 -1.85714524 -0.17879624 -2.41872763
      -1.31041083]
     [-1.21417464 -1.27915719 -0.33780363  0.94014048 -0.39399593 -0.93737192
       1.03255477 -0.75339317  0.4502701   0.77349075 -0.86904655 -0.53395584
       1.18751465  0.07209434  0.33667889  0.2477228   0.42132182  0.43610593
       0.44259921]
     [ 0.36181786 -1.10304474  0.13088332 -0.62745518  1.34885667 -0.93737192
      -0.60559119 -0.26666494 -1.04137358 -1.04991064 -0.86904655 -0.53395584
       0.06166651 -0.26911077  0.12923028 -0.35822406 -0.05377164 -0.40570396
      -0.06869538]
     [-0.65132018 -0.92693229 -1.27517753 -0.62745518  0.47743037 -0.93737192
      -0.60559119 -0.26666494 -0.29555174 -1.04991064  0.18502281 -0.53395584
      -0.24297475 -0.72405091 -0.17836593  0.43907444  0.42132182  0.21650335
       0.51271961]
     [-1.15788919 -0.75081984 -0.33780363  0.15634265  1.34885667 -0.93737192
      -0.60559119 -0.26666494  1.19609194  0.77349075  0.18502281 -0.53395584
       0.71068485  0.75450455 -0.17836593  1.58718428  1.47152842  0.43610593
       0.86332162]
     [-0.53874928 -0.57470739 -1.27517753  0.15634265 -1.26542223 -0.93737192
       0.21348179 -0.75339317 -1.04137358 -0.13820995 -0.86904655 -0.53395584
       0.06166651 -0.26911077  0.12923028 -0.35822406 -0.05377164 -0.40570396
      -0.06869538]
     [-0.70760562 -0.39859494 -1.74386449 -1.41125301  0.47743037 -0.93737192
      -0.60559119 -0.26666494 -1.04137358 -1.04991064  0.18502281 -0.53395584
       0.23385505  0.64076952  0.46544009 -0.32633212 -0.62888478 -0.27760246
       0.47765941]
     [-0.03218027 -0.22248248  0.59957028  1.72393831 -0.39399593 -0.93737192
       1.03255477 -0.75339317  0.4502701   0.77349075  0.18502281 -0.53395584
      -0.839012   -1.06525601  0.20791768 -1.47444196  0.42132182 -1.26581407
       0.37247881]
     [-1.10160375 -0.04637003 -1.27517753 -0.62745518 -0.39399593  0.09018447
       0.21348179 -0.75339317  0.4502701  -0.13820995  0.18502281  0.9314996
      -0.6005971  -1.17899105  1.62429094  0.2477228   0.72138085  0.38120529
       0.54777981]
     [ 1.43124134  0.12974242 -0.33780363 -0.62745518 -0.39399593  1.11774086
      -0.60559119  0.2200633  -1.04137358 -1.04991064 -0.86904655 -0.53395584
       0.47226995 -0.26911077 -1.07969437 -0.7090354  -0.32882575 -0.77170826
      -0.83709812]
     [-1.21417464  0.30585487 -2.21255144 -0.62745518 -1.26542223 -0.93737192
       1.03255477 -0.75339317 -1.04137358 -0.13820995  0.18502281  0.9314996
       0.23385505  0.41329944  0.7229625   1.96988756  1.92161696  0.05180142
       0.0919972 ]
     [ 0.81210143  0.48196732  0.13088332  0.15634265 -0.39399593  1.11774086
      -0.60559119  0.2200633   0.4502701  -0.13820995  0.18502281 -0.53395584
       0.71068485  1.20944469 -1.07969437 -0.51768376 -0.77891429  0.271404
       0.84579152]
     [ 1.31867045  0.65807977  0.59957028 -0.62745518  2.22028297  0.09018447
      -0.60559119 -0.26666494  0.4502701  -1.04991064  0.18502281 -0.53395584
      -0.48138965 -0.72405091  0.5942013  -0.90038704 -1.07897332 -0.38740375
      -1.04745933]
     [-0.76389107  0.83419222 -0.80649058  0.94014048 -1.26542223  0.09018447
       1.03255477 -0.75339317  0.4502701   0.77349075  0.18502281  0.9314996
       0.47226995  0.86823959  1.10924612 -0.13498048 -0.32882575  0.21650335
       0.72308082]
     [-0.82017651  1.01030467 -0.33780363  0.15634265  0.47743037 -0.93737192
       1.03255477 -0.75339317 -1.04137358 -0.13820995  0.18502281  0.9314996
       0.3530625  -0.3828458  -1.59473919 -1.09173868 -0.62888478 -0.77170826
      -0.32872521]
     [ 2.78209205  1.18641712 -0.33780363  0.15634265 -0.39399593  0.09018447
      -0.60559119  4.11388914  1.19609194  0.77349075  0.18502281 -0.53395584
       1.0683072   0.52703448 -0.69341075 -1.6657936  -3.92953409 -0.9364102
      -0.118364  ]
     [ 2.8383775   1.36252957 -0.80649058  0.15634265 -0.39399593 -0.93737192
       0.21348179 -0.75339317 -0.29555174 -0.13820995 -0.86904655 -0.53395584
      -0.3621822  -0.95152098 -0.56464955 -0.51768376  0.12126279 -0.44230439
      -0.2060145 ]
     [ 1.03724321  1.53864202 -1.74386449 -0.62745518 -0.39399593 -0.93737192
      -0.60559119 -0.26666494 -1.04137358 -1.04991064 -0.86904655 -0.53395584
      -0.48138965 -1.06525601  0.98048491 -1.28309032 -0.02876672 -0.66190697
      -0.78450782]
     [ 4.07665732  1.89086693  0.13088332 -1.41125301 -1.26542223  2.14529725
      -0.60559119  0.2200633  -1.04137358 -1.04991064 -0.86904655 -0.53395584
      -0.48138965 -1.63393119  0.46544009 -1.09173868 -0.47885526 -1.86972117
      -2.09926535]]
    X_train Type:
     <class 'numpy.ndarray'>
    X_train Shape:
     (20, 23)
    One-Hot Style Shape:
     (20, 4)
    X_test_num Shape:
     (20, 19)
    


```python
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
probabilities = log_reg.predict_proba(X_test)[:,1]
prob_table = pd.DataFrame({'year':field['year'],
                           'odds':field['odds'],
                           'probability':probabilities})
print('Probabilities:\n', prob_table.sort_values('probability', ascending=False))
```

    Probabilities:
                       year  odds  probability
    name                                     
    Taiba             2022     5     0.418214
    Cyberknife        2022    12     0.093010
    Epicenter         2022     5     0.076662
    CrownPride        2022    17     0.066775
    WhiteAbarrio      2022    13     0.064548
    ChargeIt          2022    14     0.051837
    TizTheBomb        2022    26     0.042752
    SmileHappy        2022    15     0.040832
    Messier           2022     6     0.037053
    Zandon            2022     7     0.034849
    MoDonegal         2022     9     0.023554
    SummerIsTomorrow  2022    33     0.022803
    Zozos             2022    45     0.022108
    Happy Jack        2022    20     0.020640
    PioneerOfMedina   2022    52     0.009493
    Simplification    2022    41     0.006008
    TawnyPort         2022    77     0.005339
    BarberRoad        2022    50     0.003717
    ClassicCauseway   2022    76     0.002890
    RichStrike        2022    99     0.001501
    
