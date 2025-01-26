# Group3_phase4_Project

### Group Members
- [Gideon Ochieng](https://github.com/OchiengGideon)  
- [Lorna Wangui](https://github.com/lorna-creator)  
- [Ann Mwangi](https://github.com/ann-mw)  
- [Charles Odhiambo](https://github.com/T-hoveen)  
- [Victor Masinde](https://github.com/Masinde10)  

---

## Project Overview
We recognized that the movie industry generates billions annually, but with the rise of streaming platforms, users often faced choice paralysis due to the overwhelming number of movie options. To address this, we developed a Movie Recommendation System to deliver personalized suggestions, enhancing user satisfaction and engagement.

![Movie Image](ml-latest-small/karen-zhao-jLRIsfkWRGo-unsplash.jpg)

## Business Understanding

### Objective
Our goal was to:
- Build a recommendation system to provide personalized Top-5 movie recommendations.  
- Improve user engagement for streaming platforms.  
- Help stakeholders understand user preferences to guide content strategies.  

### Stakeholders
1. Primary: End users and streaming platforms.  
2. Secondary: Movie studios and market researchers.  

---

## Project Details

We began by identifying the problem of choice paralysis in streaming platforms and defined our objectives:
- Build a recommendation system using collaborative and content-based filtering.  
- Address cold-start issues with a hybrid approach.  
- Generate insights into user preferences for stakeholders.

We used the MovieLens dataset from GroupLens research lab, which comprised:  
- Movies Dataset: Contained movie IDs, titles, and genres.  
- Ratings Dataset: Included user ratings for movies, scaled from 0.5 to 5.0.  

We explored the data to identify key attributes, check for data quality issues, and ensure suitability for building recommendation models.

### Data Preparation
We prepared the data as follows:
- Removed duplicates and handled missing values.  
- Mapped movie IDs to titles for better interpretability.  
- Normalized user ratings and created user-item matrices.  
- Generated content-based features, such as genres and keywords, for item similarity.  

We also split the data into training and testing sets to facilitate model evaluation.

### Modeling
We implemented the following models:
- Collaborative Filtering:  
  - Used K-Nearest Neighbors (KNN) with cosine similarity to recommend movies based on similar user preferences.  
  - Applied Singular Value Decomposition (SVD) for latent factor analysis and matrix factorization.  
- Content-Based Filtering:  
  - Leveraged movie attributes (e.g., genres) to compute item similarity using cosine similarity.  
- Hybrid Model:  
  - Combined collaborative and content-based filtering to address the cold-start problem for new users and items.  

### Evaluation
We evaluated the models using:
- Root Mean Squared Error (RMSE): Achieved a value of **0.87.  
- Mean Absolute Error (MAE): Achieved a value of **0.68.  

We compared the performance of different models and fine-tuned their hyperparameters to achieve optimal results.

### Deployment
We designed a simple recommendation pipeline where:
- Users provided ratings for movies they had watched.  
- The system returned a personalized list of movie recommendations.  

We proposed integrating the system into streaming platforms to provide real-world benefits.

---

## Key Results
- Our system delivered accurate and diverse recommendations.  
- The hybrid model successfully addressed the cold-start problem.  
- Evaluation metrics indicated strong performance, scalability, and relevance to user preferences.  

---

## Conclusion
We successfully built a movie recommendation system that addressed the challenge of overwhelming choices by providing personalized and efficient suggestions. This not only improved user experience but also added value for streaming platforms by increasing user engagement and retention.

---

## Recommendations
To further enhance the system, we recommend:  
1. Incorporating Real-Time Feedback: Dynamically update recommendations based on user interactions.  
2. Exploring Advanced Machine Learning Techniques: Experiment with methods like autoencoders or neural collaborative filtering to improve model accuracy.  
3. Leveraging Multi-Modal Data: Integrate additional data sources, such as trailers, reviews, or images, to enrich the recommendation process and improve diversity.
