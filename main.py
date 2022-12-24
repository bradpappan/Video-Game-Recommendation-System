import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from random import randint

df = pd.read_csv("metacritic_critic_reviews.csv", error_bad_lines=False, encoding='utf-8')
print(df.columns[df.isna().any()].tolist())
df.dropna(inplace=True)


def add_year(full_date):
    datetime_object = datetime.strptime(full_date, '%b %d, %Y')
    return datetime_object.year


df['year'] = df['date'].apply(add_year)


def year_game(row):
    calendar_year = str(row['year'])
    year_game_combined = str(row['game']) + " (" + calendar_year + ")"
    return year_game_combined


df['game'] = df.apply(year_game, axis=1)

unique_titles = df['game'].unique()
title_sum_score = {}

for i in unique_titles:
    title_sum_score[i] = 0.0


def total_score_per_title(row):
    title_sum_score[row['game']] += float(row['score'])


df.apply(total_score_per_title, axis=1)

sum_df = pd.DataFrame.from_dict(title_sum_score, orient='index', columns=['score'])
sum_df = sum_df.reset_index().rename(columns={'index': 'game'}).sort_values('score', ascending=False)

top100_sum_df = sum_df.iloc[:100]
top100_sum_df['game'].unique()

random = randint(0, int(top100_sum_df.shape[0]) - 1)
print(random)

game_name = top100_sum_df['game'].iloc[random]
print(game_name)

current_game_reviews = df.loc[df['game'] == game_name]

current_game_indexes = current_game_reviews.index

random = randint(0, int(current_game_indexes.shape[0]) - 1)

current_review_index = current_game_indexes[random]
current_review_index

profile = {'game': [],
           'name': [],
           'score': []}


def loop_10_games():
    counter = 0
    inputs = []

    print(
        "Based on the sentiment of each review quote, please score them on how relevant it is to your own perspective "
        "of that game title.")
    print("These are the scores with the corresponding responses:")
    print("""
  2 - I would rate the game MUCH better /n 
  1 - I would rate the game a little better /n 
  0 - This is spot on /n 
  -1 - I would rate the game worst /n
  -2 - I would rate the game MUCH worst /n
  N - No comment
   """)

    while counter < 10:

        random_top_100 = randint(0, int(top100_sum_df.shape[0]) - 1)

        while top100_sum_df['game'].iloc[random_top_100] in inputs:
            random_top_100 = randint(0, int(top100_sum_df.shape[0]) - 1)
        current_game = top100_sum_df['game'].iloc[random_top_100]
        current_game_reviews = df.loc[df['game'] == current_game]

        current_game_indexes = current_game_reviews.index
        random = randint(0, int(current_game_indexes.shape[0]) - 1)
        current_loc = current_game_indexes[random]
        print("Do you agree with the sentiment of this quote? Rate the relevancy between -2 to +2")
        print(df['game'].iloc[current_loc])
        print(df['review'].iloc[current_loc])

        user_response = input()
        user_responses = ["I would rate the game MUCH better", "I would rate the game a little better", "This is spot "
                                                                                                        "on",
                          "No comment", "I would rate the game worst", "I would rate the game much worst"]
        user_increments = [20, 10, 0, 'NaN', -20, -30]

        inputs.append(top100_sum_df['game'].iloc[random_top_100])

        if user_response == "1":  # "I would rate the game a little better"
            user_score = int(df['score'].iloc[current_loc]) + 10

            if user_score > 100:
                user_score == 100

            profile['game'].append(df['game'].iloc[current_loc])
            profile['name'].append(1001)
            profile['score'].append(user_score)
            counter += 1
        elif user_response == "2":  # "I would rate the game MUCH better"
            user_score = int(df['score'].iloc[current_loc]) + 20
            if user_score > 100:
                user_score == 100
            profile['game'].append(df['game'].iloc[current_loc])
            profile['name'].append(1001)
            profile['score'].append(user_score)
            counter += 1
        elif user_response == "0":  # "This is spot on"
            user_score = int(df['score'].iloc[current_loc])
            profile['game'].append(df['game'].iloc[current_loc])
            profile['name'].append(1001)
            profile['score'].append(user_score)
            counter += 1
        elif user_response == "-1":  # "I would rate the game worst"
            user_score = int(df['score'].iloc[current_loc]) - 20
            if user_score < 20:
                user_score == 20
            profile['game'].append(df['game'].iloc[current_loc])
            profile['name'].append(1001)
            profile['score'].append(user_score)
            counter += 1
        elif user_response == "-2":  # "I would rate the game much worst"
            user_score = int(df['score'].iloc[current_loc]) - 30
            if user_score < 20:
                user_score == 20
            profile['game'].append(df['game'].iloc[current_loc])
            profile['name'].append(1001)
            profile['score'].append(user_score)
            counter += 1

        elif user_response == "N":  # "No comment"
            continue
        elif user_response == "End":
            break
        else:
            continue


loop_10_games()
my_ratings = pd.DataFrame(profile)
my_ratings.head()

my_ratings

main_df = df[['game', 'name', 'score']]


def add_profile(current_df, profile_df):
    complete_df = pd.concat([current_df, profile_df], axis=0)
    complete_df.columns = ['itemID', 'userID', 'rating']
    complete_df['reviews'] = complete_df.groupby(['itemID'])['rating'].transform('count')
    return complete_df


updated_df = add_profile(main_df, my_ratings)


def pivot_data_similarity(full_df):
    pivot = full_df.pivot_table(index=['itemID'], columns=['userID'], values='rating')
    pivot_n = pivot.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=1)

    pivot_n.fillna(0, inplace=True)

    pivot_n = pivot_n.T

    pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

    piv_sparse = sp.sparse.csr_matrix(pivot_n.values)

    game_similarity = cosine_similarity(piv_sparse)

    sim_matrix_df = pd.DataFrame(game_similarity, index=pivot_n.index, columns=pivot_n.index)

    return sim_matrix_df


new_sim_df = pivot_data_similarity(updated_df)

most_similar_critics = []


def game_recommendation(reviewer):
    """
    This function will return the top 5 reviewers with the highest cosine similarity value and show their match percentage.

    """
    top_5_most_similar = []

    number = 1
    print('Recommended critics based on how similar your tastes are:')

    for n in new_sim_df.sort_values(by=reviewer, ascending=False).index[1:6]:
        top_5_most_similar.append(n)

        print("#" + str(number) + ": " + n + ", " + str(round(new_sim_df[reviewer][n] * 100, 2)) + "% " + "match")
        number += 1
    return top_5_most_similar


most_similar_critics = game_recommendation(1001)

critic_titles = df[df['name'] == most_similar_critics[0]].sort_values('score', ascending=False)


def top_critic(critic):
    number = 1
    print("These are your most similar critic\'s ({}) highest scored games:\n".format(critic))

    for n in range(len(critic_titles['game'][:10])):
        print(
            "#" + str(number) + ": " + str(critic_titles.iloc[n]['game']) + ", " + str(critic_titles.iloc[n]['score']))
        number += 1


top_critic(critic_titles.iloc[0]['name'])


def recommend_games():
    profile = {'game': [], 'name': [], 'score': []}

    loop_10_games()
    profile_df = pd.DataFrame(profile)

    full_df = add_profile(main_df, profile_df)
    new_sim_df = pivot_data_similarity(full_df)
    most_similar_critics = []
    most_similar_critics = game_recommendation(1001)
    critic_titles = df[df['name'] == most_similar_critics[0]].sort_values('score', ascending=False)

    top_critic(critic_titles.iloc[0]['name'])


recommend_games()
