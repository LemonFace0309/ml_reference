# GRADED FUNCTION: cofi_cost_func
# UNQ_C1
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    ### START CODE HERE ###  
    
    for i in range(nm):
        for j in range(nu):
            loss = R[i][j] * (np.dot(W[j], X[i]) + b[0][j] - Y[i][j])**2
            J += loss

    # Regularization
    J += lambda_ * (np.sum(np.square(W)) + np.sum(np.square(X)))
    
    J /= 2
    print(f"cost {J}")
    
    ### END CODE HERE ### 

    return J


### Vectorized Implementation
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J



### Learning movie recommendations
movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}')

# New user ratings:

# Rated 5.0 for  Shrek (2001)
# Rated 5.0 for  Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
# Rated 2.0 for  Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
# Rated 5.0 for  Harry Potter and the Chamber of Secrets (2002)
# Rated 5.0 for  Pirates of the Caribbean: The Curse of the Black Pearl (2003)
# Rated 5.0 for  Lord of the Rings: The Return of the King, The (2003)
# Rated 3.0 for  Eternal Sunshine of the Spotless Mind (2004)
# Rated 5.0 for  Incredibles, The (2004)
# Rated 2.0 for  Persuasion (2007)
# Rated 5.0 for  Toy Story 3 (2010)
# Rated 3.0 for  Inception (2010)
# Rated 1.0 for  Louis Theroux: Law & Disorder (2008)
# Rated 1.0 for  Nothing to Declare (Rien à déclarer) (2010)

# Reload ratings and add new ratings
Y, R = load_ratings_small()
Y    = np.c_[my_ratings, Y]
R    = np.c_[(my_ratings != 0).astype(int), R]

# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

## Custom training loop

# The operations involved in learning  𝑤 ,  𝑏 , and  𝑥  simultaneously do not fall into the typical 'layers' offered in the TensorFlow neural network package. Consequently, the flow used in Course 2: Model, Compile(), Fit(), Predict(), are not directly applicable. Instead, we can use a custom training loop.

# Recall from earlier labs the steps of gradient descent.

# - repeat until convergence:
# -- compute forward pass
# -- compute the derivatives of the loss relative to parameters
# -- update the parameters using the learning rate and the computed derivatives

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

# Training loss at iteration 0: 1768.5
# Training loss at iteration 20: 1768.1
# Training loss at iteration 40: 1767.8
# Training loss at iteration 60: 1767.5
# Training loss at iteration 80: 1767.2
# Training loss at iteration 100: 1767.0
# Training loss at iteration 120: 1766.8
# Training loss at iteration 140: 1766.6
# Training loss at iteration 160: 1766.4
# Training loss at iteration 180: 1766.2


### Recommendations
# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

# Predicting rating 4.49 for movie My Sassy Girl (Yeopgijeogin geunyeo) (2001)
# Predicting rating 4.48 for movie Martin Lawrence Live: Runteldat (2002)
# Predicting rating 4.48 for movie Memento (2000)
# Predicting rating 4.47 for movie Delirium (2014)
# Predicting rating 4.47 for movie Laggies (2014)
# Predicting rating 4.47 for movie One I Love, The (2014)
# Predicting rating 4.46 for movie Particle Fever (2013)
# Predicting rating 4.45 for movie Eichmann (2007)
# Predicting rating 4.45 for movie Battle Royale 2: Requiem (Batoru rowaiaru II: Chinkonka) (2003)
# Predicting rating 4.45 for movie Into the Abyss (2011)


# Original vs Predicted ratings:

# Original 5.0, Predicted 4.90 for Shrek (2001)
# Original 5.0, Predicted 4.84 for Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
# Original 2.0, Predicted 2.13 for Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)
# Original 5.0, Predicted 4.88 for Harry Potter and the Chamber of Secrets (2002)
# Original 5.0, Predicted 4.87 for Pirates of the Caribbean: The Curse of the Black Pearl (2003)
# Original 5.0, Predicted 4.89 for Lord of the Rings: The Return of the King, The (2003)
# Original 3.0, Predicted 3.00 for Eternal Sunshine of the Spotless Mind (2004)
# Original 5.0, Predicted 4.90 for Incredibles, The (2004)
# Original 2.0, Predicted 2.11 for Persuasion (2007)
# Original 5.0, Predicted 4.80 for Toy Story 3 (2010)
# Original 3.0, Predicted 3.00 for Inception (2010)
# Original 1.0, Predicted 1.41 for Louis Theroux: Law & Disorder (2008)
# Original 1.0, Predicted 1.26 for Nothing to Declare (Rien à déclarer) (2010)

filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

# pred	mean rating	number of ratings	title
# 1743	4.030965	4.252336	107	Departed, The (2006)
# 2112	3.985287	4.238255	149	Dark Knight, The (2008)
# 211	4.477792	4.122642	159	Memento (2000)
# 929	4.887053	4.118919	185	Lord of the Rings: The Return of the King, The...
# 2700	4.796530	4.109091	55	Toy Story 3 (2010)
# 653	4.357304	4.021277	188	Lord of the Rings: The Two Towers, The (2002)
# 1122	4.004469	4.006494	77	Shaun of the Dead (2004)
# 1841	3.980647	4.000000	61	Hot Fuzz (2007)
# 3083	4.084633	3.993421	76	Dark Knight Rises, The (2012)
# 2804	4.434171	3.989362	47	Harry Potter and the Deathly Hallows: Part 1 (...
# 773	4.289679	3.960993	141	Finding Nemo (2003)
# 1771	4.344993	3.944444	81	Casino Royale (2006)
# 2649	4.133482	3.943396	53	How to Train Your Dragon (2010)
# 2455	4.175746	3.887931	58	Harry Potter and the Half-Blood Prince (2009)
# 361	4.135291	3.871212	132	Monsters, Inc. (2001)
# 3014	3.967901	3.869565	69	Avengers, The (2012)
# 246	4.897137	3.867647	170	Shrek (2001)
# 151	3.971888	3.836364	110	Crouching Tiger, Hidden Dragon (Wo hu cang lon...
# 1150	4.898892	3.836000	125	Incredibles, The (2004)
# 793	4.874935	3.778523	149	Pirates of the Caribbean: The Curse of the Bla...
# 366	4.843375	3.761682	107	Harry Potter and the Sorcerer's Stone (a.k.a. ...
# 754	4.021774	3.723684	76	X2: X-Men United (2003)
# 79	4.242984	3.699248	133	X-Men (2000)
# 622	4.878342	3.598039	102	Harry Potter and the Chamber of Secrets (2002)
