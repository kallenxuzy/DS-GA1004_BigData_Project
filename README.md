# DSGA1004 - BIG DATA
## Final project
- Claire Ellison Chen, Yi Wen, Zhuoyuan Xu

*Handout date*: 2021-04-08

*Submission deadline*: 2021-05-18

# Recommender-System-DSGA1004
A recommender system built in PySpark and designed for HPC environment. Submitted as term project for DS-GA 1004 Big Data at NYU.

For complete instruction and structure of project, please refer to original project description [here](https://github.com/nyu-big-data/final-project-team_gxs/blob/main/instructions.md)

## Description of files:
 
- Hyperparameter tuning with ALS: 
  - ***sample_indexer.py***: downsample and StringIndex
  - ***learning_curve.py***: learning curve to determine training set size
  - ***param_train_1st.py***, param_train_2nd.py: parameter tuning for each parameter and for grid search
  - ***one_train.py***: training a single model with set parameters

- Single Machin (LightFM) implementation:
  - LightFM is a package for recommendation system to run on single machine
  - ***als_model_extension_2.py***: Spark ALS (rank=150, regParam=0.05, max_iter=1)
  - ***extension2_1004project.ipynb***: LightFM implementation
  
- Latent factor exploration
  - Under folder '\Extension: UMAP visualization'
  - ***Exploration-EDA.ipynb***: data cleaning and outputs df_final.csv for UMAP
  - **UMAP visualization_Final.ipynb**: UMAP parameter tuning and visualization for default ALS
  - **UMAP visualization_Final_BestALS.ipynb**: UMAP parameter tuning and visualization for best ALS
  - Data used:
  - ***Genre_10.csv***: top 13 track genres cleaned and saved from Exploration-EDA.ipynb
  - ***df_final.csv***: cleaned music metadata for UMAP visualization from Exploration-EDA.ipynb
  - ***dominant_trackgenre.csv***: cleaned list of track genre by index order of df_final.csv from Exploration-EDA.ipynb
  - ***item_matrix.csv, user_matric.csv***: ALS(10,1,1,max_iter=10) learned item factors and user factors (on 5% training data)
  - ***Item_matrix_full.csv***: best ALS model learned item factors (on 5% training data)
  - ***track_ids.csv***: string index output by ALS versus track_ids; used to join metadata with item factors

- Final report: 
  - [DSGA_1004_FInal_Report.pdf](https://github.com/nyu-big-data/final-project-team_gxs/blob/main/1004_MSD_Recommendation_System_Final_Report.pdf)

## Contributors

This project exists thanks to all the people who contribute, especially the three main authors:
- Claire Ellison-Chen: basic recommender system
- Yi Wen: metadata cleaning, UMAP parameter tuning, extension - UMAP visualization [(profile](https://www.linkedin.com/in/yi-sophia-wen/), [GitHub)](https://github.com/yiwen1996)
- Zhuoyuan Xu: parameter tuning, extension - single machine


And our professor Brian McFee who has provided advising and guidence.
