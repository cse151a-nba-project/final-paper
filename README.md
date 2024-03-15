# NBA Performance Prediction Using Machine Learning

CSE 151A Winter 2024

Ji, Daniel <daji@ucsd.edu>; Weng, Max <maweng@ucsd.edu>; Li, Aaron <all042@ucsd.edu>; Wang, David <dyw001@ucsd.edu>; Jin, Bryant <brjin@ucsd.edu>; Ye, Eric <e2ye@ucsd.edu>; Shen, Kevin <k3shen@ucsd.edu>; Sood, Roshan <rosood@ucsd.edu>; Lu, Kevin <k8lu@ucsd.edu>; Kanetkar, Dhruv <dkanetkar@ucsd.edu> 

Repo link: <https://github.com/cse151a-nba-project/final-paper>

# Abstract (Introduction)

Although sports analytics captured national attention only in 2011 with the release of Moneyball, research in the field is nearly a century old. Until relatively recently, this research was largely done by hand; however, the heavily quantitative nature of sports analytics makes it an attractive target for machine learning. This paper explores the application of advanced machine learning models to predict team performance in National Basketball Association (NBA) regular season and playoff games. Several models were trained on a rich dataset spanning over four decades, which includes advanced individual player metrics. The core of our analysis lies in combining individual player metrics without the need of team data to create machine learning models, using entirely real data that can be sufficient enough in size to predict team performance. We employ various machine learning techniques, including deep neural networks and support vector models, to generate predictive models for player performance and compare these models’ performance with both each other and traditional predictive models like linear regression. Our analysis suggests that the elastic net model method outperforms other models, with neural networks and support vector models overfitting. Moreover, we note that a multi-model approach with ensemble learning also results in a performant model, even when including less performant models such as linear regression models and overfitting deep neural networks. Our findings emphasize the immense potential of sophisticated machine learning techniques in sports analytics and mark a growing shift towards computer-aided and computer-based approaches in sports analytics.

# Project Introduction

Our project focuses on applying advanced machine learning techniques to predict team performance in the National Basketball Association (NBA). We chose this topic because of the increasing importance and popularity of sports analytics in recent years, as well as a personal interest by many of our team members.

The NBA, with its rich history and extensive data availability, provides an excellent opportunity to explore the potential of machine learning in sports analytics. By developing accurate predictive models for team performance, we can gain valuable insights that can help teams make better decisions in terms of player selection, game strategies, and overall team management.

Moreover, the broader impact of having a good predictive model for NBA team performance is significant. It can assist teams in optimizing their roster, identifying key player attributes that contribute to team success, and making data-driven decisions during the game. This can lead to improved team performance, increased fan engagement, and even financial benefits for the teams and the league as a whole.

Furthermore, the techniques and approaches used in this project can be extended to other sports and domains. The principles of applying machine learning to predict outcomes based on individual performance metrics can be applied to various fields, such as business, healthcare, and education. By demonstrating the effectiveness of these techniques in the context of the NBA, we aim to inspire further research and application of machine learning in sports analytics and beyond.


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

## Data Exploration
Data exploration results are closely related to the process of exploring the data itself, please refer to the methods section.

## Data Preprocessing and Calculation
For our first model, our data preprocessing tries to predict given the statistics of the top 8 players of a team the expected win percentage. For these statistics, we chose to use the following: 'per', 'ws_48', 'usg_percent', 'bpm', 'vorp'. These stand for player efficiency rating, win shares per 48 minutes, usage percentage, box plus/minus, and value over replacement player values. We chose these statistics for each player as we believe they holistically encompass a player's value in numbers. For instance, with player efficiency rating, we are able to cover both positive accomplishments like field goals, free throws, 3-pointers, assists, rebounds, blocks and steals, and negative ones like missed shots, turnovers and personal fouls. However, 'per' has some limitations - it doesn't account well for defensive contributions beyond steals and blocks, and can reward inefficient volume scoring. Similarly, both 'ws_48' and 'bpm' are able to further solidify an individual player's contribution to the team and exactly how many points they are putting into the game. We then tried to counteract the limitations brought up by 'per', 'ws_48', and 'bpm' using 'usg_percent' and 'vorp', which gives statistics on how the player does within the team. This allows us to also consider the synergy of this player with the rest of the team and ensure that the player is an essential part of why a team might be performing well. Hence, our first model took in a training input of shape (number of teams, 8 players * 5 statistics) = (number of teams, 40). Each of these input features of 40 is mapped to a scalar output feature representing our win percentage. This makes our training output of shape (number of teams, 1). Note that we scaled up our win percentage by 100, so instead of 38% win percentage equating to 0.38, we train it based on 38. This is done in order to have a more accurate MSE and MAE, and we found that our models still performed significantly well even with this scaling.

However, for our first model we found our results to be lacking, which is also available in our milestone 3 analysis. Thus, we decided to work with more input features to hopefully achieve more accurate measures. We had to balance between adding more features and also not having an imbalance in features which can lead to overfitting. For our models 2 and 3, we decided to take the top 10 players of a team now to predict the expected win percentage. Moreover, for our statistics, we added in four extra statistics: 'mp_per_game', 'ts_percent', 'experience', and 'x3p_ar'. These stand for minutes played per game, true shotting percentage, player experience, and three point attempts. The four new statistics - 'mp_per_game', 'ts_percent', 'experience', and 'x3p_ar' - enhance our understanding of the five previously discussed metrics ('per', 'ws_48', 'usg_percent', 'bpm', 'vorp') by providing additional context and nuance. 'ts_percent' helps to more accurately assess a player's scoring efficiency, which is a key component of 'per', 'ws_48', 'usg_percent', 'bpm', 'vorp'. For instance, someone with lower 'ts_percent' but higher 'per' can be deemed less valuable towards the overall winning percentage and vice versa depending on the general trend for players. Furthermore, experience contextualizes these metrics, as a young player posting impressive numbers is often more noteworthy than a veteran doing the same. Thirdly, 'x3p_ar' adds depth to our interpretation of 'usg_percent' and efficiency metrics, as it indicates a player's shooting style and role within the modern, three-point-oriented NBA. Finally, 'mp_per_game' is crucial for interpreting all of these statistics, as it provides insight into a player's role and the sustainability of their performance. A player with impressive per-minute numbers in limited playing time might have the potential for an even greater impact with increased minutes. Ultimately, these four statistics work in concert with the original five to provide a more comprehensive picture of a player's contributions and value. Thus, for our models 2 and 3, we took in a training input of shape (number of teams, 10 players * 9 statistics) = (number of teams, 90). Each of these input features of 90 is mapped to a scalar output feature representing our win percentage.

## Model Results

|              | Linear | Elastic Net |  DNN   | Tuned DNN | SVR model | Ensemble model |
|--------------|--------|-------------|--------|-----------|-----------|----------------|
| Training MSE | 16.704 |   17.376    | 19.954 |  11.957   | 12.083    | 14.803         |
| Training MAE | 3.2729 |   3.3292    | 3.5214 |  2.3812   | 2.134     | 2.995          |
| Training R^2 | 0.9312 |   0.9284    | 0.9178 |  0.9507   | 0.9502    | 0.9390         |
| Testing MSE  | 20.881 |   19.921    | 24.686 |  37.071   | 25.281    | 19.634         |
| Testing MAE  | 3.7103 |   3.6713    | 4.1783 |  4.7918   | 4.168     | 3.680          |
| Testing R^2  | 0.9028 |   0.9072    | 0.8850 |  0.8274   | 0.8823    | 0.9086         |

![Untitled](https://github.com/cse151a-nba-project/final-paper/assets/73797155/83cb1423-16ed-4da0-82a0-b3f375d8c95e)
![Untitled](https://github.com/cse151a-nba-project/final-paper/assets/73797155/8afe6a6f-ebde-4333-b6b8-bf73249b12da)
![Untitled](https://github.com/cse151a-nba-project/final-paper/assets/73797155/f014d5fc-c82b-4ca3-8659-33a4d4275fe3)

# Discussion

We decided to use a simple linear regression model as our first model as a proof of concept. We wanted to keep it simple at first so that we can easily pivot if needed later. Additionally, since Neural Networks are essentially combinations of linear regressions and non linearities, a simple linear regression would give us a good baseline model performance to test future models against. Our simple linear regressor enabled us to get a testing MSE of 20.881 and a testing R^2 of 0.902, which was very promising. The high R^2 score indicated that a very high percentage of the variability in the data was captured by our model. However, we hoped to be able to use more complex models to capture more details in the data to enable us to further improve our testing and training metrics.

Based on the analysis of our simple linear regressor, which indicates underfitting given the high mean squared error (MSE) rates for both testing and training, we are considering using a Deep Neural Network (DNN) model which can capture more of the intricacies in the data. We also intend to experiment with Elastic Net models that can capture more details of the data but with regularization to prevent overfitting.

Our hyperparameter-tuned DNN model showed a remarkable performance on the testing data. The training MSE has significantly decreased to approximately 8.98, indicating a much better fit to the training data. However, the testing MSE is extremely high at around 41.25. This discrepancy between training and testing errors suggests that the model might be overfitting to the training data, and not effectively generalizing to other datapoints. The hyperparameter tuning process we conducted has likely resulted in a complex model that performs exceptionally well on the training data but struggles to generalize to unseen data. The R^2 value for training is impressively high at 0.96, while the testing R^2 is lower at 0.81. This further supports the notion of overfitting.

Our Elastic Net model, on the other hand, showed a marginal improvement in performance compared to our linear regression model. The training MSE has decreased to approximately 17.38, while the testing MSE is around 19.92. This indicates a better fit to the data compared to our initial model. The training and testing errors are relatively close, suggesting that the model is not overfitting or underfitting, and it might be approaching the best fit point of the curve The Elastic Net regularization we applied has effectively balanced the model's complexity and generalization ability. The R^2 values for both training and testing are above 0.90, indicating a strong correlation between the predicted and actual win percentages.

We then decided to try using a Support Vector Regressor which we hypothesized would be able to capture the non-linearities in the data without overfitting to the training data. Unfortunately, after training and tuning, our SVR model was overfitting and we decided that it would likely not improve significantly more with feature expansion. We can see this because the SVR model has training MSE of and testing MSE of 12 and 25 (relatively far apart, with testing greater, suggesting that the model is overfitting), while our best model, the elastic net model (which is likely near the best fit region), has a training MSE of 17 and testing MSE of 20. 

It seems that we are near a lower boundary of 15 MSE, where models cannot fit below without overfitting, possibly trying to analyze patterns / predict randomness. Even compared to the original linear regression model (which is also likely near the best fit region), our SVR model is worse (a 25 testing MSE vs. 21 testing MSE), indicating that the model complexity is too high and at the overfitting stage. Nevertheless, the R^2 values for both training and testing for the SVR model are above 0.88, indicating a strong correlation between the predicted and actual win percentages.

For the Ensemble model, testing MSE is 20 and R^2 coefficient is 0.91, which is relatively the same as the elastic net model performance: likely in the best fit region of the fitting graph. Combining all models' predictions together and averaging them results in likely the best model, better or equal than any individual model. Nevertheless, the metrics are about the same as the elastic net model.

# Conclusion
In this paper, we explored the application of advanced machine learning techniques to predict team performance in the NBA. By leveraging a dataset spanning over four decades and including advanced individual player metrics, we developed several models to predict team win percentages based on the performance of their top players. Our analysis involved extensive data exploration, preprocessing, and calculation to ensure the most relevant features were used in our models. We experimented with linear regression, elastic net, deep neural networks (DNN), support vector regression (SVR), and ensemble models. The elastic net model emerged as the best performer, while the DNN and SVR models showed a tendency to overfit. The ensemble model, which combined all the individual models, also demonstrated strong performance. Our findings highlight the potential of machine learning in sports analytics and the importance of careful feature selection and model tuning. By focusing on individual player metrics, we demonstrated that it is possible to predict team performance without relying on team-level data. This approach can be particularly useful for teams looking to optimize their roster and identify key player attributes that contribute to success. However, our study also underscores the challenges of applying machine learning in complex domains like sports. The risk of overfitting, the need for extensive data preprocessing, and the importance of domain knowledge in feature selection are all important considerations.

Future work in this area could take several exciting directions by incorporating additional data sources and techniques to improve the predictive power of the models. One promising avenue is the integration of player tracking data. With the advent of advanced camera systems and wearable technology, the NBA now collects highly granular data on player movements, such as speed, distance covered, and the location of shots. This data could provide valuable insights into player efficiency, off-ball movement, and defensive impact, which are not fully captured by traditional box score statistics. By incorporating this information, future models could paint a more comprehensive picture of individual player contributions and their impact on team success.Another area worth exploring is the inclusion of health and fitness data. Injuries and player fatigue can significantly impact team performance, and by integrating data on player health, such as minutes played, injury history, and load management, future models could better account for the impact of player availability on team success. This could be particularly valuable for teams looking to optimize player rotations and minimize the risk of injury. Off-court factors, such as player morale, team chemistry, and public perception, can also influence team performance. Analyzing social media data, such as player tweets or fan sentiment, could provide additional context and help predict how these factors might affect on-court performance. While this data may be more subjective and harder to quantify, it could still offer valuable insights into the intangible aspects of team dynamics. Expanding the scope of the data to include information on opposing teams and specific matchups could also improve the predictive power of the models. By incorporating data on opposing teams' defensive ratings, playing styles, or head-to-head records, future models could better account for the impact of specific matchups on game outcomes. This could be particularly useful for coaches and analysts looking to develop game plans and strategies tailored to specific opponents. Coaching strategies, offensive and defensive schemes, and player rotations also play a significant role in team performance. Future work could explore the incorporation of data on coaching tendencies, such as substitution patterns or the frequency of certain play calls, to better capture the impact of coaching on team success. This could provide valuable insights for teams looking to optimize their coaching strategies and player rotations. Finally, techniques like transfer learning could be applied to leverage insights from other sports or domains. Models trained on player or team data from sports such as soccer or hockey might provide valuable approaches or insights that could be adapted to NBA data. This could potentially accelerate the development of more accurate and comprehensive models for predicting NBA team performance.

In conclusion, there are numerous exciting opportunities for future work in this area. By incorporating additional data sources, such as player tracking data, health and fitness information, social media sentiment, opponent and matchup data, coaching tendencies, and historical data, researchers could develop even more powerful and insightful models for predicting NBA team performance. However, it is crucial to carefully evaluate the relevance and quality of any additional data before incorporating it into the models to ensure that it contributes meaningfully to the predictive power and interpretability of the results. We believe our research demonstrates the immense potential of machine learning in sports analytics and highlights the growing importance of data-driven decision making in the NBA and beyond. As the field continues to evolve, we expect to see even more sophisticated techniques and applications emerge, transforming the way we analyze and understand sports performance.

# Collaboration

#### Daniel Ji: Project Manager

Contribution: Oversaw project and worked and verified work on majority of project aspects: dataset exploration, data exploration, individual model creation and hyperparameter tuning, data visualization, and data interpretation and writeup. 


#### Max Weng: Data Visualization & Model Engineer

Contribution: Helped create various data visualizations to analyze linear regression model, elastic net model, DNN model, and SVR model performance. Also worked on coding and tuning the models themselves. Worked on writing final research paper.


#### Aaron Li: Model Engineer

Contribution: Worked on the coding of linear regression model, elastic net model, DNN model, and SVR models and integrating data preprocessing code to provide model input data. Worked on writing final research paper.


#### David Wang: Data Preprocessing Engineer

Contribution: Preprocessed data to be scaled / normalized / feature extracted so that it can be used as input data for models. Also assisted in the initial data exploration / visualization process. Worked on writing final research paper.


#### Bryant Jin: Data Exploration Engineer

Contribution: Worked with Eric in taking raw dataset and determined interesting features while also providing visualizations for data distribution / type / correlation. Was more involved in finding potentially robust and useful NBA datasets and doing data exploration.


#### Eric Ye: Data Exploration Engineer

Contribution: Worked with Bryant in taking raw dataset and determined interesting features while also providing visualizations for data distribution / type / correlation. Was more involved in taking potentially robust NBA datasets from Bryant and working with David to preprocess the data (connecting model creation steps as a middle man).


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

Milestone 5 Submission: <https://github.com/cse151a-nba-project/milestone-5>

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
