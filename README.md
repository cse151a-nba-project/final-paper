# NBA Performance Prediction Using Machine Learning

CSE 151A Winter 2024

Ji, Daniel <daji@ucsd.edu>; Weng, Max <maweng@ucsd.edu>; Li, Aaron <all042@ucsd.edu>; Wang, David <dyw001@ucsd.edu>; Jin, Bryant <brjin@ucsd.edu>; Ye, Eric <e2ye@ucsd.edu>; Shen, Kevin <k3shen@ucsd.edu>; Sood, Roshan <rosood@ucsd.edu>; Lu, Kevin <k8lu@ucsd.edu>; Kanetkar, Dhruv <dkanetkar@ucsd.edu> 


# Abstract (Introduction)

# Project Introduction

# Previous Submissions

See Appendix for all previous submissions and their corresponding Github repositories (which include corresponding jupyter notebooks).


# Methods

## Data Exploration

### Data Interpretation (What the Data Means)

The Kaggle Dataset we are using is a **very comprehensive dataset** with many features, spanning from individual player stats to team performance measurements for nearly the entirety of the NBA. Here is a breakdown of some of the features we found interesting while exploring our data:


#### Across all CSVs

All-season related data files contain season identification (what season, what type of league), all team-related data files contain team identification (team name, abbreviation), and all player-related data files contain player identification and other information (name, age, position, etc.). Because players may have the same name, we can use the player_id column to distinguish.\
**Regarding team data files**, although we are predicting team performance based on the players on that team, this team data would be possibly useful in scaling the players performance accordingly depending on their team. For example, a player may not necessarily have a good plus/minus, but that might be because their team overall is not a good team and the overall negative team performance affects their own performance. Nevertheless, we still focused on taking a deeper dive into player stats more than team stats when exploring data. 


#### Advanced.csv

Some of the more complex measurements for a player’s performance. Using these stats, which are feature-engineered from more simple stats (points, rebounds, assists, minutes played, etc.), we can get a more holistic measurement of individuals. Interesting columns: 

1. **per**: Player efficiency rating, a measure of per-minute production **standardized** such that the league average is 15. 

2. **usg_percent**: Usage Percentage, an estimate of the percentage of team plays used by a player while they were on the floor. **Normalized** from 0 to 1 since it’s a percentage. 

3. **ws** and **ws_48**: Win Shares, an estimate of the number of wins contributed by a player. Also exists WS/48 which is win shares per 48 minutes, a **scaled** feature. 

4. **bpm**: Box Plus/Minus, a box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team (**scaled**).

5. **vorp**: Value over Replacement Player, a box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.


#### All-Star Selections.csv

Lists all-star selections, voted by current NBA players and fans. After reviewing all data features from this dataset, although award selection likely has a strong correlation with player performance, we decided to look more into the numerical objective statistics and exclude awards and subjective data features (based on fellow player / media perception) like this data file from our project (all these files would make our project scope too large, this dataset is huge!).


#### End of Season Teams.csv, End of Season Teams (Voting).csv

Lists the best players, in their respective positions (30 players total by season), voted by the media. For similar reasons to All-Star voting, we did not decide to use this data. 


#### Per 100 Poss.csv, Per 36 Minutes.csv, Player Totals, Player Per Game.csv

These datafiles hold the same columns, with slight differences but with Per 100 Poss.csv being the most fit since it is both normalized to the other stats (Per 100 Poss. for teams and opposing teams) and contains the most information (also has offensive and defensive ratings of players). Here are the key features of interest, both raw and slightly feature-engineered: 

1. **fga** and **fg_percent**: field goals (total attempts) and their accuracy (how many they made / how many they attempted). Excluded field goals because it can be derived from the two easily (will do so for all metrics like this). 

2. **x2pa** and **x2p_percent**, **x3pa** and **x3p_percent**: same as fga and fga_percent, but instead just two / three pointers 

3. **e_fg_percent**: effective field goal percentage 

4. **trb** (total rebound percentage)

5. and raw stats: **ast** (assists), **stl** (steals), **blk** (blocks), **tov** (turnovers), **pts** (points)


#### Player Award Shares.csv

Player awards (like end of season teams voting, all-star selections, MVP, rookie of the year, defensive player of the year, most improved player, etc.). Decided not to use after reviewing all files, see All-Star Selections.csv for elaboration.


#### Player Career Info.csv

Lists the first, last, and number of seasons that an NBA player was active. Note that if a NBA player was not playing in the NBA for some seasons, but returned later, the number of seasons may be less than the number of seasons between the first and last season the NBA player was active. 


#### Player Season Info.csv

List of NBA players and age, birth year, team, position, and experience (how many years they’ve played) for every season. Like Player Career Info.csv, is helpful supplemental information that can be possible features for calculating team composition / chemistry (how long teammates have stuck together). The two features can possibly used in feature engineering to develop a score of how long a team has stuck together and how much players on the team have played as teammates. 


#### Player Play by Play.csv

List of player stats, related to the player’s actions / performance based on possessions (plays).

1. **g** and **mp**: games and minutes played, useful for potential standardization of these metrics. 

2. **on_court_plus_minus_per_100_poss**, **net_plus_minus_per_100_poss**: metric to measure change score (own team scored points - opponent team scored points) while each player is on the court (and also off the court for net plus/minus). Useful for measuring holistic player performance (not just raw features like points, rebounds, assists, etc. stats)

3. **points_generated_by_assists**: points generated by assist, all points for which the player gets an assist for


#### Player Shooting.csv

Detailed player shooting statistics. Many of these columns are definitely too specific for predicting team performance from player stats and wouldn’t be used, but some generalized ones could be interesting to model with:

1. **fg_percent**: a raw feature of the field goal percentage of a player

2. **avg_dist_fga**: the average distance of a field goal attempt - a combination of far and close shooters (guards and conventional forwards/centers) or just far shooters (guards and forwards/centers that can also shoot from far away OR small-ball team with guards) could possibly result in a better team than other combinations


#### Team Stats Per 100 Poss.csv, Team Stats Per Game.csv, Team Totals.csv

Also includes many key features (nearly all raw features) about teams. The raw features are as follows (per 100 possessions, as named; stats per game and totals are excluded because they are more or less the same with just different values)

1. **fga_per_100_poss** and **fg_percent** (see above)

2. **x3pa_per_100_poss** and **x3p_percent** (see above)

3. **x2pa_per_100_poss** and **x2p_percent** (see above)

4. **trb_per_100_poss** (true rebounding) 

5. **ast_per_100_poss** (assists)

6. **stl_per_100_poss** (steals)

7. **blk_per_100_poss** (blocks)

8. **tov_per_100_poss** (turnovers)

9. **pts_per_100_poss** (points)


#### Team Summaries.csv

Includes both raw and statistically calculated metrics about teams, where the statistically calculated metrics may give a better measure of the team than actual results, since they attempt to reduce luck and other random factors. 

1. **w** and **l**, **pw** and **pl**:

2. **mov**: margin of victory,

3. **o_rtg, d_rtg, n_rtg**: net rating,

4. **e_fg_perrcent** and **opp_e_fg_percent**: effective field goal percentage  


#### Opponent Stats Per 100 Pos.csv, Opponent Stats Per Game.csv, Opponent Totals.csv

1. **opp_fga_per_100_poss** and **fg_percent** (see above)

2. **opp_x3pa_per_100_poss** and **x3p_percent** (see above)

3. **opp_x2pa_per_100_poss** and **x2p_percent** (see above)

4. **opp_trb_per_100_poss** (true rebounding) 

5. **opp_ast_per_100_poss** (assists)

6. **opp_stl_per_100_poss** (steals)

7. **opp_blk_per_100_poss** (blocks)

8. **opp_tov_per_100_poss** (turnovers)

9. **opp_pts_per_100_poss** (points)


#### Summary

Here are the files we are potentially interested in for modeling: 

1. Advanced.csv

2. Player Play by Play.csv

3. Player per Game.csv

4. Per 100 Pos.csv + Player Shooting.csv (can be merged in one)

5. Team Stats per 100 Poss.csv

6. Team Summaries.csv

7. Opponents Stats Per 100 Poss.csv


### Data Visualizations

Visualizations from: <https://github.com/cse151a-nba-project/milestone-2/blob/main/CSE_151A_Milestone_2_Per_100_Poss_csv_(Players)%2C_Player_Shooting_csv.ipynb> (which is combined from <https://drive.google.com/drive/u/2/folders/1CH3FFtG6fX7vUAg5j7ONCgpFA2h3ZBmV>). 

May have to download to properly view the notebook and see full resolution images. 

Note that visualizations below are not all visualizations, just some highlighted ones to show the variety of types and scope of data visualizations that we explored.


####

#### Advanced Player Data, Pairplot, Colored By Decade

Note that these features are largely what we decided to use for our models’ input data. 

![](https://lh7-us.googleusercontent.com/kkE21YBqpWccSOQDZpGKrg33qs1BTX8FUemsqmOuagT_tLBSFTLtA0OwXvBltcVUclhia5S4k0SMU6B97vM0NVfgGhcQEiFzzzxD5JKwx3cdb7DD0rBxhElgKJRt6i6iznLl2ITilgaebIDuhpSTsZU)


#### Advanced Player Data, QQ Plots to Test Normal Distribution

Note that these features are largely what we decided to use for our models’ input data.

![](https://lh7-us.googleusercontent.com/Q7fyEgGzrYPtA8YgpcyXHmtuYfwaB2fQhAB13HT2EZCrC-KA_nwV6GNv32tiCOJsUnR4-DRzj15E_P7KFs1svw1IMbxcvk6cU2hMKHx_-s_ZH4V-jYkzzVvu580W9sY1ZT8DopB3vGpKguRYOZQ5uuw)


####

#### Advanced Player Data, Correlation Coefficient Matrix

#### ![](https://lh7-us.googleusercontent.com/6oKHTMegLMd7TSthM3VqBf5fJKUT6w7VckFL1M8-vyfqsTrvPsQb1i-HKlG5H8TnyfkFoOuK_A7cqAAg0M3AGE3MYuEoGBx75A6B-Pq9YLF18iTqgQhOdh1JgsdygAVd8HXViGhsROq7J8hlmjloIIE)

#### Player Stats Play By Play, Pairplot

#### ![](https://lh7-us.googleusercontent.com/xMwxNH-Ixaz3dtRokj_xE0F28UpLAnlS1_dA0jp-tAakAUanBUrpkQ-03ozwBqgN_FOOO6u90Qlm4Av9I8CpxxaoO7t920PTrkIRttdzDjWNw5OKgr0HpSxtauut3N_dzcXYt9Y7UOn1Bk8FYkVr8Xg)

#### Player Stats Play By Play, Coefficient Correlation Matrix

#### ![](https://lh7-us.googleusercontent.com/8X0BIFC-GNSI-6QZzqCzpo9SYa4stKhr8eY3vzbOgQ4tS81n_N8PKAhjdbD8dPjUPWLLuIw90h_yuMOcYIeG6ViozT9IlkshZ30_0t1r9ClklfBtygtG8rAPwAIKb1TQZ3G_2FkX326gFgZ1M7LdX6I)

#### Player Stats Play By Play, QQ Plots![](https://lh7-us.googleusercontent.com/fWsuioXCpo8M8TCDg0E9-yitzLYq9RNfKsPx3rDsVcTVYz0sU5GIFWEiNkmj1HzMchnAAFKlov43EnxJoG8dHIUcWdzk4ZGx36DK7hYZt-B-mQxva-BZc_v1mLl7aLjoMl8pF1Oy_xGBYZK9T_gYlbw)

#### Player Stats Per Game, Comprehensive Pairplot, Colored By Decade![](https://lh7-us.googleusercontent.com/K8wwoQX3GUuOFvNDKJYnhDA2YTVWlugbAQkmr0xCioispc9tnAyeBdRMSBSOpJjMFQFnTBrVALytDcsCJI7-6XksJjQECM3RjnO68HPDEUWBAP4r_cQwFplmrOHJa1vaAcpN1UfNqw2fihJqp_DobdM)

#### Player Stats Per Game, Comprehensive Coefficient Correlation Matrix![](https://lh7-us.googleusercontent.com/2sA4bmsx0F3oETz7eBD2kvG4Ax5VorNbJdAjhbDPuIZjQEbf2lXkDPOeamO10s_NxOE4dp84Akz0svLUIcrwAcoe8Pivg8g6Oli06bj66zA-d8G5mUIxKQqYElqalwweiaQyy8pBoxl4Ur7B5RcFOY0)

#### Opponent Stats Per 100 Pos., Comprehensive Pairplot![](https://lh7-us.googleusercontent.com/lklN0OFWpsPkw8RXobVQ1phC16H-4jaCyyNZLlSYIjSEB-d8L5fzfvb9u44DnBkHxMQjt9IOHg5xeMtEL_yROSmhW9vlEB-l0yjDctCGe_F35drgcrfNFz7xnJXrjPWMuAiS127f6_6SxE2sIZMrsBk)

#### Opponent Stats Per 100 Pos., Comprehensive Coefficient Correlation Matrix

#### ![](https://lh7-us.googleusercontent.com/4AQgdgwWf9IeE1qqQF7DlWlmcPBFr7ksdILSpnBFesRwND67vjsPMuH9aPBUyDRjmjutIiJbeB9BcKhx3VS7CeP4w4-IGNqotfJhSj_qqT7bghscfhQrlnnERUFks8EfPJLAxC4OkBTuYaFAA0x9fhY)

## Data Preprocessing and Calculation

### Missing Data

Regarding missing columns or data, **given that we are looking into NBA data after 1990**, the data files were well filled out and comprehensive (please see below for pre-1990 NBA data). Please read more below regarding the entire dataset (since the start of the NBA). Because there are so many players in the NBA and it would be quite hard to check for missing rows, we generally assumed that the ~11,000 filtered NBA player data observations (post-1990) were accurate. We assume likewise that NBA team data post-1990 was accurate and complete as well.  With columns, we checked for NaN values and dropped rows that had them (there were very few instances, if not none, for our data files, so this wouldn’t skew our data in any way). 

For example, in the player Advanced.csv, we ran: 
```python
print(advanced_player_df_stats.isna().values.any())
print(advanced_player_df_stats.isnull().values.any())
```
Where advanced_player_df_stats contained 'decade', 'experience', 'per', 'usg_percent', 'ws_48', 'bpm', 'vorp'. The output for both lines was false, indicating we likely had no missing data. 

Looking at the dataset as a whole, imputations came from many different fields from our ~35,000+ player data entries being filled with NaN values. This is because in our combined database, much of the past seasons did not record certain attributes that we were intending to use for our machine learning. Thus, after filtering out a bit of the NaN's we found that we can discard seasons 1947-1973, as they did not record any useful data such as assist percentage, steal percentage, etc. Thus, shortening the dataset around then still kept us with around ~25,000+ player data entries, which would be sufficient for our machine learning algorithm. Some attributes were only being recorded starting very recently, which led us to removing these options completely. For example, Games Started was only recorded starting 1982, which we found to be not worth keeping as it would limit more of our dataset for an unneeded value.


### Filtering Data: Cumulative Past ~25 Season Player Data

In addition to filtering to look only into post-1990 games and players that only played 40+ games in a specific season, we removed many of the categorical variables except for their positions, as we believed they would just hinder our machine learning algorithm's ability to interpret the data on its own. Additionally, we would look for outliers in our data (using the 1.5IQR from first and third quartile rule) to remove outliers. Sometimes, data would be distributed more widely, and so we would instead manually remove extreme (irregular) values. 


### Data Calculation: Top X Players From A Team In A Given Season

Since we intend to use player stat predictions to predict team success, we needed a way to compare different players across different teams. We hypothesized that since only a certain number of players will play in any given game, the most relevant players to analyze would be the top 8 players for each team (note that we later changed this to 10). We determined that minutes played would be the best metric to determine the top players for each team, since playing more minutes would naturally mean a greater impact on a team’s performance. 

We used the Player per Game CSV file. To find the top players for each team, we first isolated the player data for a specific season. We chose the 2023 season, since it’s the most recently completed season, and top players change every year. Then we sorted the dataframe by minutes played. We extracted the list of teams and for each team, we extracted the top players with the most minutes played from that team. 

We then generated a pairplot showing the relationships between the stats of the top player and win-loss percentage for each team, and then we looked at the pairplot for the 8th best player, to see how depth plays a role. Please see the notebook for code and visualizations: <https://drive.google.com/file/d/1AkgBmlF83gzglRQ-r8pjKyTVGZBi3IQX/view?usp=sharing>.

Note that we ended up altering the function as we began model training to better adapt to the models and input data. For the final code function we used to extract top player data by minutes played, see <https://colab.research.google.com/drive/1VhsZ8wjTWmwQclFHv-5u3dM4crS6UJp3#scrollTo=jnSz2UsFCyK2&line=2&uniqifier=1> (labeled “Milestone 4 code, needed for model building in this milestone”). 


### Normalization, Standardization, and Data Transformation

Much of the dataset had normalized features - per 100 possessions, being a percentage, the NBA metric was within a certain range or scaled, etc. Along the way, we standardized / normalized / scaled data. Please see the notebook for these calculations. 


### Data Encoding

We one-hot-encoded player positions to better supply data to any models for this project. For the most part, we did not encode most variables, since the value we are trying to predict, win-loss percentage, is already a number. Many of the input statistics / features are also numerical, so at most we’ll just have to normalize or standardize. We may need to one-hot encode some numerical data into categories / brackets, but that also likewise does not require data encoding.


## Model 1: Linear Regression Model

See milestone 3 for model and write-up: <https://github.com/cse151a-nba-project/milestone-3>

Core model code:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(all_data_x, all_data_y, test_size = 0.2, random_state=21)
regressor = LinearRegression().fit(X_train, y_train)
```

## Model 2: Elastic Net Model & DNN Model

See milestone 4 for model and write-up: [https://github.com/cse151a-nba-project/milestone-4](https://github.com/cse151a-nba-project/milestone-4/)

### Elastic Net Model
Core model code blocks:

```python
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
scaled_all_data_x = scaler_x.fit_transform(all_data_x)
scaler_y = MinMaxScaler()
scaled_all_data_y = scaler_y.fit_transform(all_data_y.reshape(-1, 1))  # Reshape if necessary
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(scaled_all_data_x, scaled_all_data_y, test_size=0.2, random_state=21)
```

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline

elastic_net_model = make_pipeline(MinMaxScaler(), ElasticNet(random_state=21))

param_grid = {
    'elasticnet__alpha': np.linspace(0.001, 0.02, num=20),
    'elasticnet__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1] # Mix ratio between Lasso and Ridge, 1 is Lasso, 0 is Ridge
}

# Grid search to find best parameters
grid_search = GridSearchCV(elastic_net_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_parameters = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best parameters found: ", best_parameters)

# Best parameters from GridSearchCV
best_alpha = best_parameters['elasticnet__alpha']
best_l1_ratio = best_parameters['elasticnet__l1_ratio']

# Setting up the ElasticNet regressor with the best parameters
elastic_net_regressor = make_pipeline(MinMaxScaler(), ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=21))

# Training the regressor with your training data
elastic_net_regressor.fit(X_train, y_train)
```

### DNN Model:
Core model code blocks:

```python
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV, KFold
from scikeras.wrappers import KerasRegressor

# Define the model function
def create_model(hidden_layers, initial_units, activation):
    model = Sequential()
    model.add(Dense(initial_units, input_dim=X_train.shape[1], activation=activation,
                    kernel_regularizer=l1_l2(0.001)))
    units = initial_units
    for _ in range(hidden_layers - 1):
        units //= 3
        model.add(Dense(units, activation=activation,
                        kernel_regularizer=l1_l2(0.001)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# Define the hyperparameters to tune
param_grid = {
    'model__hidden_layers': [3, 4, 5, 6],
    'model__initial_units': [30, 90, 270, 810],
    'model__activation': ['relu', 'tanh'],
    'batch_size': [8, 32]
}

# Create the model wrapper
model = KerasRegressor(model=create_model)

# Define the GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True),
    scoring='neg_mean_squared_error',
    return_train_score=True,
    n_jobs=-1,
    verbose=0
)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# Retrieve the best model
best_model = grid_search.best_estimator_

# Set early stopping callback
early_stopping = EarlyStopping(monitor='val_mse', patience=25, restore_best_weights=True)

# Fit the best model with early stopping
history = best_model.fit(X_train, y_train, epochs=500, validation_split=0.2, callbacks=[early_stopping], verbose=0)
```

```python
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping

# Define the model
manual_model = Sequential([
    Dense(270, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l1_l2(0.001)),  # First hidden layer with regularization
    Dense(90, activation='relu', kernel_regularizer=l1_l2(0.001)),
    Dense(30, activation='relu', kernel_regularizer=l1_l2(0.001)),  # Second hidden layer with regularization
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
manual_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Fit the manually tuned model
history = manual_model.fit(X_train, y_train, epochs=500, batch_size=8, validation_split=0.2, callbacks=[early_stopping])
```

## Model 3: SVR Model & Ensemble Model

See milestone 5 for model and write-up: <https://github.com/cse151a-nba-project/milestone-5>
### SVR Model
Core model code blocks:

```python
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

# Define the hyperparameters to tune
param_grid = {
    'kernel': ['poly', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1],
    'gamma': ['scale', 'auto'] + list(np.logspace(-2, 2, 4)),
    'degree': [2, 3, 4]
}

# Create the SVR model
model = SVR()

# Define the GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

# Retrieve the best model
best_svr_model = grid_search.best_estimator_

# Fit the best model
best_svr_model.fit(X_train, y_train)
```

### Ensemble Model
Core model code blocks:

```python
def ensemble_predict(X):
  ensemble_models = {
    'Elastic net model': elastic_net_regressor,
    'Linear regression model': regressor,
    'Manual DNN': manual_dnn_model,
    'HP-Tuned DNN': best_model,
    'SVR Model': best_svr_model
  }
  overall_prediction = None
  for model_name, model in models.items():
    prediction = model.predict(X).flatten()
    if (overall_prediction is None):
      overall_prediction = prediction
    else: 
      overall_prediction += prediction
  return overall_prediction / len(models.items())
```

# Results

# Discussion

# Conclusion

# Collaboration

#### Daniel Ji: Project Manager

Contribution: Oversaw project and worked and verified work on majority of project aspects: dataset exploration, data exploration, individual model creation and hyperparameter tuning, data visualization, and data interpretation and writeup. 


#### Max Weng: Data Visualization & Model Engineer

Contribution: Helped create various data visualizations to analyze linear regression model, elastic net model, DNN model, and SVR model performance. Also worked on coding and tuning the models themselves. 


#### Aaron Li: Model Engineer

Contribution: Worked on the coding of linear regression model, elastic net model, DNN model, and SVR models and integrating data preprocessing code to provide model input data. 


#### David Wang: Data Preprocessing Engineer

Contribution: Preprocessed data to be scaled / normalized / feature extracted so that it can be used as input data for models. Also assisted in the initial data exploration / visualization process. 


#### Bryant Jin: Data Exploration Engineer

Contribution: Worked with Eric in taking raw dataset and determined interesting features while also providing visualizations for data distribution / type / correlation. Was more involved in finding potentially robust and useful NBA datasets and doing data exploration.


#### Eric Ye: Data Exploration Engineer

Contribution: Worked with Bryan in taking raw dataset and determined interesting features while also providing visualizations for data distribution / type / correlation. Was more involved in taking potentially robust NBA datasets from Bryant and working with David to preprocess the data (connecting model creation steps as a middle man).


#### Kevin Shen: Model Engineer

Contribution: Worked closely with Aaron to create various optimal machine learning models (linear regression, elastic net, DNN, SVR) and hyperparameter tune and analyze and improve performance. 


#### Roshan Sood: Data Preprocessing Engineer

Contribution: Explored all sorts of features / data entries from datasets and gave context behind them, as one of the project members more knowledgeable in basketball. Looked at raw data manually to potentially spot interesting features and look for missing data / inconsistencies.


#### Kevin Lu: Documentation & Write-Up 

Contribution: Wrote analysis and documentation for jupyter notebooks and general project milestone deadlines. 


#### Dhruv Kanetkar: Documentation & Write-Up

Contribution: Wrote analysis and documentation for jupyter notebooks and general project milestone deadlines.


# Appendix

Dataset: <https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats>

Milestone 1 Submission: <https://docs.google.com/document/d/1-GX9r1rib8BSJ24MR7iWZP2uWUlcFml0J-0UAyBK_Sk/edit?usp=sharing>

Milestone 2 Submission: <https://github.com/cse151a-nba-project/milestone-2>

Milestone 3 Submission: <https://github.com/cse151a-nba-project/milestone-3>

Milestone 4 Submission: <https://github.com/cse151a-nba-project/milestone-4>


# References

During the creation of our models, we used a variety of online resources to guide our development. We primarily used the following sites:

<https://www.tensorflow.org/tutorials>

<https://keras.io/guides/>

<https://github.com/keras-team/keras-tuner/issues>

<https://scikit-learn.org/stable/index.html>

<https://numpy.org/doc/>

<https://pandas.pydata.org/docs/>

<https://www.geeksforgeeks.org/>

<https://stackoverflow.com/>
