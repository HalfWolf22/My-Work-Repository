This is a (maybe not so) little project that I did in my spare time between learning theory. Here, I would like to
present my Business Intelligence skills I could deliver in a real-world example project (as of 16. February 2020).
I made all of this from scrap, from scraping data, cleaning/wrangling, visualizing, modifying, exploring, plotting
and making predictive models.

The README file should contain all of the additional information about this repository you are going to need to
better understand or navigate it. All of the datasets are not included in the repository because of their size,
so you have 2 options:
	1. You could run all the scraping scripts and notebooks and then arrange the datasets into their respective 
	folders where they belong (see the section under the line down). (P.S. Game and Player datasets have to be
	scrapped in the exact chunks as I did to avoid bugging out the notebook since I didn't use the OS library+
	for loop to take all the datasets from the folder but rather hard codded them in.)
	2. I will provide a MediaFire link in the root directory of this project where you can download the whole
	 if you want to save time and of course if you don't mind downloading something off of some random
	link I gave you. This also saves you of the hustle of having to move the datasets to the folders where they
	are supposed to be since you are downloading everything already prepared.

	Link: http://www.mediafire.com/file/58nidymdaxh5v1e/Robbing_the_NBA.zip/file

Feel free to use the datasets from this repository just don't forget to give the credit!

Enjoy! :)


*easydatascience is a custom library with which I operate faster. I ignored it because even if it is found in some
directories, it might not be used in all of them.

--------------------------------------------------------------------------------------------------------------------

--DATA SCRAPING--

-Divided into game scraper and player scraper because different pages were used, thus different scrappers were
needed.
   

1.Game Scraper
     |
     |--- game_... CSVs - Laptop is crappy so I scraped the data in chunks if something fails
     |
     |--- game_scraper - Game data scraping script, chunk functionality added for the reason above
     |
     |--- Merge and Clean Up - A bit of data manipulation so it is more useful
               |
               |--- game_data.csv - The exported file that is located in 3.Getting Useful Data   

2.Player Scraper
     |
     |--- 2.a Player Heights
                 |
                 |--- height_scraper - They had to be separately scraped since the heights weren't included in
                 |                     the page for player scraping but it seems like a useful stat
                 |
                 |--- heights - Raw scraped data
                 |
                 |--- Clean Up - Nothing special
                         |
                         |--- player_heights.csv - The exported file that is located in 2.Player Scraper
     |
     |--- nba_... - The data is more massive than game data so there are more chunks, again, scrappy laptop
     |
     |--- player_scraper - Player data scraping script, chunk functionality added for the reason above
     |
     |--- player_heights - Used to add additional info to data
     |
     |--- Merge and Clean Up - Converting date column and adding heights, exporting away_table
            |
            |--- away_table.csv - The exported file that is located in 1.Game Scraper
            |
            |--- player_data.csv - The exported file that is located in 3.Getting Useful Data

--DATA MANIPULATION--

-Used for various cleaning and data mining.

3.Getting Useful Data
    |
    |--- away_table - Used for labeling away/home teams. It is an external data frame derived from player data
    |                 since the web page from which the data was scraped didn't contain this information
    |
    |--- player_data - Scraped player data
    |
    |--- game_data - Scraper game data
    |
    |--- Preparing Vanilla DataFrame - Final data cleaning and mining, adding top players features
           |
           |--- teams.csv - Exported file located in 6.Predictions and 5.EDA
           |
           |--- players.csv - Exported file located in 6.Predictions and 5.EDA

4.Merge and Forms
    |
    |--- players - Players dataframe
    |
    |--- teams - Teams dataframe
    |
    |--- Forms and Merge - Used to merge top players and teams dataframe which before this notebook only
         contained the names of the top players but not the stats. Also creating forms which are used for ML
            |
            |--- merged.csv - Exported file containing full data ready to do whatever, located in the root dir
            |                 and 5.EDA
            |
            |--- forms.csv - Exported file located in 6.Predictions

--USING THE DATA--

-Deriving something from data with EDA and ML algorithms


5.EDA
   |
   |--- teams.csv, eda_forms.csv - Dataframes used for analysis, merged dataframe not used to preserve other
   | 				   player's stats
   |
   |--- EDA - Analysis of the different aspects that are important for predictive models.

6.Predictions
     |
     |--- forms.csv, forms4.csv, forms3.csv, forms6.csv - Dataframe used for modeling
     |
     |--- Betting Lines Scraper
            |
            |--- lines_scraper - Scraping script (Website: https://sportsdatabase.com/)
       	    |
	    |--- lines.csv - Raw scraped data
            |
            |--- Clean Up - Cleaning up data
                    |
                    |--- lines_data.csv - Exported file located in 6.Predictions
     |
     |--- lines_data.csv - Clean lines data intended to boost ML models
     |
     |--- Predictions - Predicting the outcome of future NBA games.
            |
            | --- eda_forms.csv - Exported file located in 5.EDA. Used for EDA (duh).

--OTHER--

- merged.csv - The most complete data that can be used to derive a new dataset (like the one I did for predicting),
	       or for its' own EDA or something else (it is really the starting point).
- README


--------------------------------------------------------------------------------------------------------------------

Explanation of column abbreviation for people who don't watch basketball.
Repeating columns are not mentioned more than 1 time.


1.Game Scraper:
	- game_...csv: raw data, doesn't contain columns yet
        	- away_table.csv:
        	- Game - Unique game ID (2 is default 2nd and 3rd characters are the season, last 5
                      			 characters are the serial number of the game in that season
        	- Team - Team name abbreviation
        	- Home/Away - Marks the away team on a certain game

2.Player Scraper:
	- nba_...csv: raw data, doesn't contain columns yet
    	- player_heights.csv: 
        	- Season - Season, detones the year when the season began (thus season 96/97 -> 96)
        	- Player - Player name and surname
        	- Height (in) - Player height, expressed in inches

    	2.a Player Heights:
        	- heights.csv: raw data, doesn't contain columns yet

3.Getting Useful Data:
    	- game_data.csv:
        	- Date - The date of the game
        	- Q1 - Points that team scored in the first quarter of the game
        	- Q2 - Points that team scored in the second quarter of the game
        	- Q3 - Points that team scored in the third quarter of the game
        	- Q4 - Points that team scored in the fourth quarter of the game
        	- Total - Total points scored by the team
        	- Home/Away - Marks the away team on a certain game
        	- Won - Binary value, says if the team won the game
        	- WonQ1 - Binary value, says if the team won the first quarter
        	- WonH1 - Binary value, says if the team won the first half
    	- palyer_data.csv:
        	- Min - Minutes player spent on the court (playtime)
        	- FGM (Field goals made) - Number of shots player made shooting from the field
        	- FGA (Field goals attempted) - Number of shots player attempted from the field
        	- FG% (Field goal percentage) - Percentage of made field shots (FGM/FGA)
       	 	- 3PM (Three points made) - Number of shots player made shooting behind the arc
        	- 3PA (Three points attempted) - Number of shots player attempted shooting behind the arc
        	- 3P% (Three-point percentage) - Percentage of made three-point shots (3PM/3PA)
        	- FTM (Free throws made) - Number of shots player made from the free-throw line
        	- FTA (Free throws attempted) - Number of shots player attempted from the free-throw line
        	- FT% (Free throw percentage) - Percentage of made free throws (FTM/FTA)
        	- OREB (Offensive rebounds) - Number of offensive rebounds player grabbed 
        	- DREB (Defensive rebounds) - Number of defensive rebounds player grabbed 
        	- REB (Rebounds) - Number of total rebounds player grabbed
        	- AST (Assists) - Number of assistances player got
        	- STL (Steals) - Number of intersected balls player took from opposing team
        	- BLK (Blocks) - Number of shoots that player blocked
        	- TOV (Turnovers) - Number of turnovers player committed
                                    (Turnover - Loosing the ball and thus the possession before a player makes
                                     a shot at opposing teams basket)
       		- PF (Personal fouls) - Fouls player committed 
        	- PTS (Points) - Total points that player has scored
        	- +/- (Plus-minus) - Team difference in points while the player was on the court
    	- top_players.csv:
        	- PER (Player efficiency rating) - Not real PER, but a good substitution to determine the score
                               a player has on a team
        	- Season_Half - Half of the season (96/97 season, 1996 = half 1, 1997 = half 2)

4.Merge and Forms:
    	-players.csv: no new features, same as player_data.csv
    	-teams.csv: 
		- Player_1-5 : Ordinal number refers to the player power ranking on a team (Player_1 is the best
                               on a team, Player_2 is second best, etc.)


5.EDA:

6.Predictions:
   	 -forms.csv:
        	- Season_Wins - Wins team has accumulated this season at this date
        	- PlayedHome - Number of games teams has played on the home court in last 5 games

        	*To spare myself of redundant typing, I am just going to write completely new features. Features in
                this data frame refer to the form in the last 5 games, thus 3PM_team is the average number of 
		three's a team attempted in the last 5 games, same for player stats. For more, refer to 
                Forms and Merge notebook located in 4.Merge and Forms.
