from . import munge
from hmmlearn import MultinomialHMM

"""
Each of sequential(), nonsequential(), and baseline() take:
    - time_series: a pandas data_frame representing a series of days
        in a particular community area preprocessed for that algorithm
    - day: a datetime specifying the day being predicted

Each returns:
    - classification: True if crime is predicted. Else False.
    - probability: a value between 0 and 1 that represents the predicted probability of violent crime
"""


def sequential(time_series, day): 
    #play around with number of components- see what is most accurate
    x = get_previous_month(time_series,day) #x is the timeseries of the three previous days
    x = x[0].values.tolist()
    results = []
    #run this nine (dont have to worry about ties) times to account for the randomness- can also play around with this number
    for ind in range(0,9):
        model = MultinomialHMM(n_components=3,n_iter=10000) #initialize the model
        model.fit([np.array(x)]) #fit the model
        hidden_states = model.predict(x) #determine the hidden states for the sequence
        last_state_probs = model.predict_proba(x)[len(x)-1] #get the most recent hidden state probabilities
        current_state = np.where(last_state_probs == max(last_state_probs))[0][0] #determine the most likely current state from those probs
        transition_probs = model.transmat_[current_state] #get the probabilities of the next state given that state
        next_state = np.where(transition_probs==max(transition_probs))[0][0] #get the next state as the most likely of these probs
        emissions = model.emissionprob_[next_state] #get the emission probabilities of the current state
        output = np.where(emissions==max(emissions))[0][0] #determine the most likely of these emissions
        results.append(output) #add this output to our results array
    if np.count_nonzero(results) >4:
        return 1
    else:
        return 0
    


def nonsequential(time_series, day):
    pass


def baseline(time_series, day):
    pass


"""
Each of sequential_preprocess(), nonsequential_preprocess(), baseline_preprocess() take:
    - master_area_dict: the canonical mapping from community area names
        to pandas data frames representing a series of days in that area.
        Also includes data frame for all of Chicago.
        See munge.py for more documentation.
Each returns:
    - [sequential, nonsequential, baseline]_dict: mapping from community area names
        to pandas data frame ready for dumping into the proper algortihm
"""


def sequential_preprocess(master_area_dict):
    #Drop the unnecessary columns of each neighborhoods data frame
    sequential_dict = {}
    for key in master_area_dict.keys():
        pre_process_df = master_area_dict[key]
        pre_process_df = pre_process_df.drop(['Arrest','Domestic','Severe Crimes','Minor Crimes','Petty Crimes','Month','Weekday'],1)
	#Convert all violent crime numbers greater than 0 to 1
        index = 0
        while index < len(pre_process_df[0]):
            if pre_process_df[0][index] > 0: 
                pre_process[0][index] = 1
            index = index + 1
			
        sequential_dict[key]=pre_process_df
    return sequential_dict


def nonsequential_preprocess(master_area_dict):
    pass


def baseline_preprocess(master_area_dict):
    pass
