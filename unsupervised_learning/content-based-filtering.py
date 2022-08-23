# GRADED_CELL
# UNQ_C1
### Neural Network for content-based filtering
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###   
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_outputs)
    ### END CODE HERE ###  
])

item_NN = tf.keras.models.Sequential([
    ### START CODE HERE ###     
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_outputs)
    ### END CODE HERE ###  
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = Model([input_user, input_item], output)

model.summary()

# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    ### START CODE HERE ###     
    d = np.sum((a - b)**2)
    ### END CODE HERE ###     
    return (d)



### Content-based filtering with a neural network

# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
scaledata = True  # applies the standard scalar to data if true
print(f"Number of training vectors: {len(item_train)}")

pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
# [user id]	[rating count]	[rating ave]	Act ion	Adve nture	Anim ation	Chil dren	Com edy	Crime	Docum entary	Drama	Fan tasy	Hor ror	Mys tery	Rom ance	Sci -Fi	Thri ller
# 2	16	4.1	3.9	5.0	0.0	0.0	4.0	4.2	4.0	4.0	0.0	3.0	4.0	0.0	4.2	3.9
# 2	16	4.1	3.9	5.0	0.0	0.0	4.0	4.2	4.0	4.0	0.0	3.0	4.0	0.0	4.2	3.9
# 2	16	4.1	3.9	5.0	0.0	0.0	4.0	4.2	4.0	4.0	0.0	3.0	4.0	0.0	4.2	3.9
# 2	16	4.1	3.9	5.0	0.0	0.0	4.0	4.2	4.0	4.0	0.0	3.0	4.0	0.0	4.2	3.9
# 2	16	4.1	3.9	5.0	0.0	0.0	4.0	4.2	4.0	4.0	0.0	3.0	4.0	0.0	4.2	3.9

pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
# [movie id]	year	ave rating	Act ion	Adve nture	Anim ation	Chil dren	Com edy	Crime	Docum entary	Drama	Fan tasy	Hor ror	Mys tery	Rom ance	Sci -Fi	Thri ller
# 6874	2003	4.0	1	0	0	0	0	0	0	0	0	0	0	0	0	0
# 6874	2003	4.0	0	0	0	0	0	1	0	0	0	0	0	0	0	0
# 6874	2003	4.0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
# 8798	2004	3.8	1	0	0	0	0	0	0	0	0	0	0	0	0	0
# 8798	2004	3.8	0	0	0	0	0	1	0	0	0	0	0	0	0	0


### Preparing the training data

# scale training data
if scaledata:
    item_train_save = item_train
    user_train_save = user_train

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    print(np.allclose(item_train_save, scalerItem.inverse_transform(item_train)))
    print(np.allclose(user_train_save, scalerUser.inverse_transform(user_train)))

item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)


### Predictions

## Calculating v_m
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = Model(input_item_m, vm_m)                                
model_m.summary()

## Calculating v_ms
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])

## Calculating matrix of the squared distance between each move feature vector and all other movie feature vectors
count = 50
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    genre1,_  = get_item_genre(item_vecs[i,:], ivs, item_features)
    genre2,_  = get_item_genre(item_vecs[min_idx,:], ivs, item_features)

    disp.append( [movie_dict[movie1_id]['title'], genre1,
                  movie_dict[movie2_id]['title'], genre2]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"])
table

# movie1	genres	movie2	genres
# Save the Last Dance (2001)	Drama	John Q (2002)	Drama
# Save the Last Dance (2001)	Romance	Saving Silverman (Evil Woman) (2001)	Romance
# Wedding Planner, The (2001)	Comedy	National Lampoon's Van Wilder (2002)	Comedy
# Wedding Planner, The (2001)	Romance	Mr. Deeds (2002)	Romance
# Hannibal (2001)	Horror	Final Destination 2 (2003)	Horror
# Hannibal (2001)	Thriller	Sum of All Fears, The (2002)	Thriller
# Saving Silverman (Evil Woman) (2001)	Comedy	Cats & Dogs (2001)	Comedy
# Saving Silverman (Evil Woman) (2001)	Romance	Save the Last Dance (2001)	Romance
# Down to Earth (2001)	Comedy	Joe Dirt (2001)	Comedy
# Down to Earth (2001)	Fantasy	Haunted Mansion, The (2003)	Fantasy
# Down to Earth (2001)	Romance	Joe Dirt (2001)	Romance
# Mexican, The (2001)	Action	Knight's Tale, A (2001)	Action
# Mexican, The (2001)	Comedy	Knight's Tale, A (2001)	Comedy
# 15 Minutes (2001)	Thriller	Final Destination 2 (2003)	Thriller
# Heartbreakers (2001)	Comedy	Animal, The (2001)	Comedy
# Heartbreakers (2001)	Crime	Charlie's Angels: Full Throttle (2003)	Crime
# Heartbreakers (2001)	Romance	Stepford Wives, The (2004)	Comedy
# Spy Kids (2001)	Action	Lara Croft: Tomb Raider (2001)	Action
# Spy Kids (2001)	Adventure	Lara Croft: Tomb Raider (2001)	Adventure
# Spy Kids (2001)	Children	Princess Diaries, The (2001)	Children
# Spy Kids (2001)	Comedy	Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)	Comedy
# Along Came a Spider (2001)	Action	Swordfish (2001)	Action
# Along Came a Spider (2001)	Crime	Swordfish (2001)	Crime
# Along Came a Spider (2001)	Mystery	Ring, The (2002)	Mystery
# Along Came a Spider (2001)	Thriller	Signs (2002)	Thriller
# Blow (2001)	Crime	Training Day (2001)	Crime
# Blow (2001)	Drama	Training Day (2001)	Drama
# Bridget Jones's Diary (2001)	Comedy	Super Troopers (2001)	Comedy
# Bridget Jones's Diary (2001)	Drama	Others, The (2001)	Drama
# Bridget Jones's Diary (2001)	Romance	Punch-Drunk Love (2002)	Romance
# Joe Dirt (2001)	Adventure	Charlie's Angels: Full Throttle (2003)	Action
# Joe Dirt (2001)	Comedy	Dr. Dolittle 2 (2001)	Comedy
# Joe Dirt (2001)	Mystery	Doom (2005)	Horror
# Joe Dirt (2001)	Romance	Down to Earth (2001)	Romance
# Crocodile Dundee in Los Angeles (2001)	Comedy	Heartbreakers (2001)	Comedy
# Crocodile Dundee in Los Angeles (2001)	Drama	Scary Movie 4 (2006)	Horror
# Mummy Returns, The (2001)	Action	Swordfish (2001)	Action
# Mummy Returns, The (2001)	Adventure	Rundown, The (2003)	Adventure
# Mummy Returns, The (2001)	Comedy	American Pie 2 (2001)	Comedy
# Mummy Returns, The (2001)	Thriller	Star Trek: Nemesis (2002)	Thriller
# Knight's Tale, A (2001)	Action	Mexican, The (2001)	Action
# Knight's Tale, A (2001)	Comedy	Mexican, The (2001)	Comedy
# Knight's Tale, A (2001)	Romance	Bruce Almighty (2003)	Romance
# Shrek (2001)	Adventure	Monsters, Inc. (2001)	Adventure
# Shrek (2001)	Animation	Monsters, Inc. (2001)	Animation
# Shrek (2001)	Children	Monsters, Inc. (2001)	Children
# Shrek (2001)	Comedy	Monsters, Inc. (2001)	Comedy
# Shrek (2001)	Fantasy	Monsters, Inc. (2001)	Fantasy
# Shrek (2001)	Romance	Monsoon Wedding (2001)	Romance
# Animal, The (2001)	Comedy	Heartbreakers (2001)	Comedy
