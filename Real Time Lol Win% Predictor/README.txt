INTUITION

The idea behind this project was to make a model that can predict the probabilities that each team has of
winning a League of Legends game. The introductory portion of the EDA sums everything up nicely so I am not going
to go into much detail here. If you go through all the files contained in the project, you could get a pretty good
understanding of the whole project without a need for any additional explanation in the README file. 

*easydatascience is a custom library with which I operate faster. I ignored it because even if it is found in some
directories, it might not be used in all of them.

--------------------------------------------------------------------------------------------------------------------

1. Scraper
      |
      |--- scraper_matches.py - Scraping recent matches' IDs, no tweaking needed
      |
      |--- matches.csv - Matches scraped from scraper_matches.py
      |
      |--- scraper_games.py - Scraping games' stats from matches' IDs
                 |
		 |--- games20k.csv, games20k40k.csv, games40k60k.csv - Scraped CSV data in chunks, because the API
		      						       has a limit so the script spends the most of
								       its time sleeping; Even though I had around 
								       600k match IDs, ~60k matches was sufficient

		 NOTE: There is less than 60k matches because not all of them were properly scraped, but as
                 I said the results were sufficient. Data located in 2. EDA

2. EDA
    | 
    |--- games... .csv - Game data
    |
    |--- ... .jpg - The picture file located in the 2. EDA
    |
    |
    |--- EDA.ipynb - EDA, contains more info about the project
             |
             |--- bool_team_data.csv - Data containing only Bool values (tried to make early predictions with a 
	     |			       simple model)
	     |
	     |--- vanilla_data.csv - The whole "vanilla" data
	     |
             |--- compact_data.csv - Vanilla data with trimmed feature space based on domain familiarity
             |
             |--- diff_team_data.csv - Modified vanilla data that includes subtracted features (Blue - Purple team)
             |
             |--- diff_data.csv - Difference data but with players. Significantly smaller because player roles are
	     |     	          are distorted in ~50% of the total data so only a fraction of the sample space
	     |			  can be shaped into this sort of data. (Depreciated, redundant)
	     |
             |--- compact_diff_data.csv - Difference data but only with features from compact_data.csv

	     NOTE: Data located in 3. Predictions

3. Predictions
       |
       |--- ... .csv - Datasets used for making the predictive models
       |
       |--- sample_game.csv - Sample time-series data for final model evaluation
       |
       |--- Models.ipynb - ML
                 |
                 |--- logistic_regression.pkl - Exported Logistic Regression model
                 |
                 |--- scaled_logistic_regression.pkl - Exported Logistic Regression model, trained on scaled samples

		 NOTE: Pickle files located in 4. Demo

4. Demo
     |
     |--- demo_game.csv - Sample time-series data for final model evaluation
     |
     |--- ... .pkl - Saved models 
     |
     |--- preds.csv - Final prediction output
     |
     |--- demo_video.mp4 - Demo video with a game, demonstrating the predictive power of the model
     |
     |--- Demo Predictions.ipynb - Final model evaluation notebook

     
NOTE/CONCLUSION:
	The final objective of the project was not to predict the winning team based on the stats from the
	game that has already been ended but to give the probabilities that each team has of winning during the 
	game. The weights got suitably adjusted, so even though the final objective is not the same, it is similar
	enough to yield satisfactory results.
