import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Concatenate
import pdb

def generate_real_samples(dataset, num_samples):
    """Randomly select real samples from dataset

    Labels will always be 1 for real samples

    Args:
        dataset (tf.DataSet or numpy.array): The real training samples
        num_samples (int): How many samples to pull

    Returns:
        tuple(X_train, Y_train)
    """
    index = np.random.randint(0, dataset.shape[0], num_samples)
    X = dataset[index]
    y = np.ones((num_samples, 1))
    return X, y

def generate_latent_points(latent_dim, cat_dim, num_continuous, num_samples):
    """Get random input to generator

    Input is made up of random normal noise and a 1 hot encoded vector
        with length cat_dim

    Args:
        latent_dim (_type_): _description_
        cat_dim (_type_): _description_
        num_samples (_type_): _description_

    Returns:
        z_input: np.array
        cat_codes: Needed as a label for the Q network
    """
    # z_latent = np.random.normal(loc=0.0, scale=2, size=(num_samples, latent_dim))
    # cat_codes = np.random.randint(0, cat_dim, num_samples)
    # cat_codes = to_categorical(cat_codes, num_classes=cat_dim)
    # z_input = np.hstack((z_latent, cat_codes))
    # return [z_input, cat_codes]
    # create noise input
    noise = tf.random.normal([num_samples, latent_dim])
    # Create categorical latent code
    label = tf.random.uniform([num_samples], minval=0, maxval=10, dtype=tf.int32)
    label = tf.one_hot(label, depth=cat_dim)
    # Create one continuous latent code
    contin_codes = tf.random.uniform([num_samples, num_continuous], minval=-1, maxval=1)
    z_input=Concatenate()([label, contin_codes, noise])

    return z_input, label, contin_codes


def generate_fake_samples(generator, latent_dim, num_cat, num_continuous, num_samples):
    """Using the tf.generator network and noise, generate samples

    Args:
        generator (tf.keras.models.Model): The generator network
        latent_dim (int): Length of pure noise part of generator input
        num_cat (int): Number of categories for the latent variable
        num_samples (int): Number of samples

    Returns:
        tuple(X_train, y_train): Training data with y labels always as 0
    """
    z_input, _, _ = generate_latent_points(latent_dim, num_cat, num_continuous, num_samples)
    images = generator.predict(z_input)
    y = np.zeros((num_samples, 1))
    return images, y

def generate_all_cat_fake_samples(generator, latent_dim, cat_dim):
    """Using the tf.generator network and noise, generate samples for all variations of cat

    Args:
        generator (tf.keras.models.Model): The generator network
        latent_dim (int): Length of pure noise part of generator input
        num_cat (int): Number of categories for the latent variable
        num_samples (int): Number of samples

    Returns:
        tuple(X_train, y_train): Training data with y labels always as 0
    """
    num_samples = cat_dim
    # Use a larger scale for the noise
    # z_latent = np.random.normal(loc=0.0, scale=2, size=(num_samples, latent_dim))
    # cat_codes = np.arange(cat_dim)
    # cat_codes = to_categorical(cat_codes, num_classes=cat_dim)
    # z_input = np.hstack((z_latent, cat_codes))
    # images = generator(z_input, training=True)
    # return images
    # create noise input
    noise = tf.random.normal([num_samples, latent_dim],stddev=2.0)
    # Create categorical latent code
    label = tf.random.uniform([num_samples], minval=0, maxval=10, dtype=tf.int32)
    label = tf.one_hot(label, depth=cat_dim)
    # Create one continuous latent code
    c_1 = tf.random.uniform([num_samples, 1], minval=-1, maxval=1)

    z_input=Concatenate()([label, c_1, noise])

    images = generator(z_input, training=True)
    return images

def generate_ordered_latent_codes(span_type, cat_info, other_cat_fixed_val, target_cat, num_samples_per_cat, contin_info, other_continous_fixed_val, target_continuous, num_samples_per_continuous, latent_noise_dim=2):
    """Generate latent points in an ordered way to understand their effect on GAN output

    total_latent_code_len = latent_noise_dim + sum(cat_info) + num_continuous_codes 

    Example Input for a continuous span:
    span_type = "span_continuous"
    contin_info = [3, 10, 2]                            # 3 Continous variable from uniform ranges [-3, 3], [-10, 10], [-2, 2]
    target_continuous = 0                               # Span the 0th continuous latent code
    other_continous_fixed_val = [None, None, 1]         # The other continuous values not being spanned will be randomly sampled except last one fixed at 1
    num_samples_per_continuous = 10                     # Take 10 samples from the 0th continuous variable so evenly spaced [-3, 3]
    cat_info = [3, 3, 2]                                # 3 Categorical variables with number of elements in each category
    other_cat_fixed_val = [0, 2, None]                  # The fixed category values. None will be uniformly sampled


    Example Input for a categorical span:
    span_type = "span_categorical"
    target_categorical = 0                              # Span the 0th categorical latent code
    num_samples_per_cat = 10                            # Take 10 samples from each categorical value
    cat_info = [3, 3, 2]                                # 3 Categorical variables with number of elements in each category
    other_cat_fixed_val = [0, 2, None]                  # The fixed category values. None will be uniformly sampled
    contin_info = [3, 10, 2]                            # 3 Continous variable from uniform ranges [-3, 3], [-10, 10], [-2, 2]
    other_continous_fixed_val = [None, 2.2, 3.3]        # The other continuous values will be randomly sampled for the first and others fixed

    Example Input for a span_both span:
    span_type = "span_both"
    target_categorical = 0                              # Span the 0th categorical latent code
    target_continuous = 0                               # Span the 0th continuous latent code
    num_samples_per_cat = 10                            # Take 10 samples from each categorical value
    num_samples_per_continuous = 10                     # Needs to be equal to num_samples_per_cat
    contin_info = [3, 10, 2]                            # 3 Continous variable from uniform ranges [-3, 3], [-10, 10], [-2, 2]
    other_continous_fixed_val = [None, None, 1]         # The other continuous values not being spanned will be randomly sampled except last one fixed at 1
    cat_info = [3, 3, 2]                                # 3 Categorical variables with number of elements in each category
    other_cat_fixed_val = [0, 2, None]                  # The fixed category values. None will be uniformly sampled

    Args:
        span_type (str): Type of variable span. Can be any of ["span_categorical", "span_continuous", "span_both"]
        cat_info (list(int)): Each element in the list represents the number of categories in that categorical code
        other_cat_fixed_val (list(int)): The fixed values for the other categorical variables. If None they will be random.
        target_cat (int): Index of the target categorical code we are spanning
        num_samples_per_cat (int): Number of samples for each category of the target cat
        contin_info list(float): Each element x specifies that continuous variable is assumed to 
            be on the range of [-x, x]
        other_continous_fixed_val list(float):  The fixed values for the other continuous variables. 
            If None they will be random on the range specified by contin_info
        target_continuous int: Index of the target continuous code we are spanning
        num_samples_per_continuous (_type_): _description_
        latent_noise_dim (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
        if type == "span_categorical"
            latent_codes.shape = (num_cat_in_target, num_samples_per_cat, total_latent_code_len)
        elif type == "span_continuous"
            latent_codes.shape = (num_samples_per_continuous, 1, total_latent_code_len)
        elif type == "span_both"
            latent_codes.shape = (num_cat_in_target, num_samples_per_continuous, total_latent_code_len)
    """

    if span_type == "span_categorical":
        num_cat_in_target = cat_info[target_cat]
        # If you want to span categorical, you need to have some fixed continous codes anyway. Could be random or fixed
        spanned_cat = generate_ordered_latent_cat_codes(
            cat_info, other_cat_fixed_val, target_cat, num_samples_per_cat)
        num_samples = num_samples_per_cat*num_cat_in_target
        fixed_continuous = generate_ordered_latent_continous_codes(
            contin_info, other_continous_fixed_val, -1, num_samples)
        fixed_continuous = fixed_continuous.reshape(
            (num_cat_in_target, num_samples_per_cat, len(contin_info)))
        latent_codes_no_noise = np.concatenate(
            [spanned_cat, fixed_continuous], axis=2)
        #latent_codes_no_noise.shape = (num_cat_in_target, num_samples_per_cat, latent_code_len_no_noise)

    elif span_type == "span_continuous":
        # If you want to span continous, you need to have some fixed categorical codes anyway. Could be random or fixed
        spanned_continuous = generate_ordered_latent_continous_codes(
            contin_info, other_continous_fixed_val, target_continuous, num_samples_per_continuous)
        spanned_continuous = spanned_continuous.reshape(
            (num_samples_per_continuous, 1, len(contin_info)))
        fixed_cat = generate_ordered_latent_cat_codes(
            cat_info, other_cat_fixed_val, -1, 1, num_samples_per_continuous)
        latent_codes_no_noise = np.concatenate(
            [fixed_cat, spanned_continuous], axis=2)
        #latent_codes_no_noise.shape = (num_samples_per_continuous, 1, latent_code_len_no_noise)

    elif span_type == "span_both":
        # If you want to span both, Only 1 sample per each category, but multiple while spanning continuous
        assert num_samples_per_continuous == num_samples_per_cat
        spanned_cat = generate_ordered_latent_cat_codes(
            cat_info, other_cat_fixed_val, target_cat, num_samples_per_cat)
        continous_span_for_each_cat = np.stack([generate_ordered_latent_continous_codes(contin_info, other_continous_fixed_val, target_continuous, num_samples_per_continuous)
                                                for _ in range(cat_info[target_cat])])
        latent_codes_no_noise = np.concatenate(
            [spanned_cat, continous_span_for_each_cat], axis=2)
        #latent_codes_no_noise.shape = (cat_info[target_cat], num_samples_per_continuous, latent_code_len_no_noise)

    # Add the pure noise part of the latent code
    latent_noise = np.random.normal(loc=0, scale=1, size=(
        latent_codes_no_noise.shape[0], latent_codes_no_noise.shape[1], latent_noise_dim))

    latent_codes = np.concatenate(
        [latent_noise, latent_codes_no_noise], axis=2)
    return latent_codes

def generate_ordered_latent_cat_codes(cat_info, other_cat_fixed_val, target_cat, num_samples_per_cat):
    """Generate a span of categorical latent codes

    Example Output for
    cat_info = [3, 3, 2]
    other_cat_fixed_val = [0, None, 1]
    target_cat = 0
    num_samples_per_cat = 3

            < targ >  < cat2 ><cat3>   
    array([[[1, 0, 0, 1, 0, 0, 0, 1],   targ = 0, cat2 random, cat3 fixed at 1
            [1, 0, 0, 0, 0, 1, 0, 1],   targ = 0, cat2 random, cat3 fixed at 1
            [1, 0, 0, 1, 0, 0, 0, 1]],  targ = 0, cat2 random, cat3 fixed at 1

           [[0, 1, 0, 1, 0, 0, 0, 1],   targ = 1, cat2 random, cat3 fixed at 1
            [0, 1, 0, 0, 0, 1, 0, 1],   targ = 1, cat2 random, cat3 fixed at 1
            [0, 1, 0, 1, 0, 0, 0, 1]],  targ = 1, cat2 random, cat3 fixed at 1

           [[0, 0, 1, 0, 1, 0, 0, 1],   targ = 2, cat2 random, cat3 fixed at 1
            [0, 0, 1, 1, 0, 0, 0, 1],   targ = 2, cat2 random, cat3 fixed at 1
            [0, 0, 1, 0, 0, 1, 0, 1]]]  targ = 2, cat2 random, cat3 fixed at 1

    Args:
        cat_info (list(int)): Each element in the list represents the number of categories in that categorical code
        other_cat_fixed_val (list(int)): The fixed values for the other categorical variables. If None they will be random.
        target_cat (int): Index of the target categorical code we are spanning
        num_samples_per_cat (int): Number of samples for each category of the target cat

    Returns:
        np.array: dimensions are ((num_target_cat, num_samples_per_cat, sum(cat_info)) and all values are 1 or 0
    """

    # This first block is to accomodate spanning the categorical variable or keeping it constant when the continuous one is spanned
    if num_samples is None:
        num_samples = num_samples_per_cat*cat_info[target_cat]
        num_target_cat = cat_info[target_cat]
    else:
        num_target_cat = num_samples

    assert len(cat_info) == len(other_cat_fixed_val)

    nums_to_join = []
    for i, (len_cat, fixed_val) in enumerate(zip(cat_info, other_cat_fixed_val)):
        if i == target_cat:
            # Span all values of target category num_samples of times
            nums_to_join.append(np.repeat(np.identity(len_cat), num_samples_per_cat, axis=0))
        else:
            if fixed_val is not None:
                assert fixed_val < len_cat
                fixed_cat_codes = np.zeros((num_samples, len_cat))
                fixed_cat_codes[:, fixed_val] = 1
                nums_to_join.append(fixed_cat_codes)
            else:
                # random cat_codes
                a = np.random.choice(range(len_cat), size=num_samples)
                rand_cat_codes = np.zeros((num_samples, len_cat))
                rand_cat_codes[np.arange(num_samples), a] = 1
                nums_to_join.append(rand_cat_codes)

    spanned_cat = np.concatenate(nums_to_join, axis=1)
    spanned_cat = spanned_cat.reshape((num_target_cat, num_samples_per_cat, sum(cat_info)))
    return spanned_cat.astype(np.int32)


def generate_ordered_latent_continous_codes(contin_info, other_continous_fixed_val, target_continuous, num_samples_per_continuous):
    """_summary_

    - Assume all continuous variables a uniforly distributed on the range [0, 1]

    Example Input:
    contin_info = [3, 10, 2]
    other_continous_fixed_val = [0, None, 0]
    target_continuous = 0
    num_samples_per_continuous = 10

    Example Output:

    array([[-3.        ,  0.33984327,  0.        ],
            [-2.3333333 , -6.1514435 ,  0.        ],
            [-1.6666666 , -6.228636  ,  0.        ],
            [-1.        ,  6.900511  ,  0.        ],
            [-0.33333334,  4.7600408 ,  0.        ],
            [ 0.33333334,  2.40507   ,  0.        ],
            [ 1.        ,  6.2062693 ,  0.        ],
            [ 1.6666666 , -4.180024  ,  0.        ],
            [ 2.3333333 ,  0.02777555,  0.        ],
            [ 3.        , -7.1847234 ,  0.        ]], 

    - First column spans the target continuous variable evenly
    - Second column is random uniform sampling from second continuous variable
    - Third column is a fixed continous variable

    Args:
        contin_info list(float): Each element x specifies that continuous variable is assumed to 
            be on the range of [-x, x]
        other_continous_fixed_val list(float):  The fixed values for the other continuous variables. 
            If None they will be random on the range specified by contin_info
        target_continuous int: Index of the target continuous code we are spanning
        num_samples_per_continuous
    """

    num_samples = num_samples_per_continuous

    assert len(contin_info) == len(other_continous_fixed_val)

    nums_to_join = []
    for i, (contin_range, fixed_val) in enumerate(zip(contin_info, other_continous_fixed_val)):
        if i == target_continuous:
            # Span all values of target_continuous num_samples of times
            nums_to_join.append(np.linspace(-contin_range, contin_range, num_samples).reshape((num_samples, 1)))
        else:
            if fixed_val is not None:
                nums_to_join.append(fixed_val*np.ones((num_samples, 1)))
            else:
                # random continuous
                nums_to_join.append(np.random.uniform(low=-contin_range, high=contin_range, size=(num_samples, 1)))

    spanned_contin = np.concatenate(nums_to_join, axis=1)
    return spanned_contin.astype(np.float32)