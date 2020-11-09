import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

def get_user_movieId():
    df = pd.read_csv('E:/data_movie/ml-20mnew/ratings.csv', header=0)
    groups = df.groupby(by='userId')
    users=df['userId'].drop_duplicates()
    #print(users)
    print(len(users))

    groups = df.groupby(by='userId')
    for i in groups:
        print(i[0])
        i[1].sort_values('timestamp').iloc[0:round(len(i[1].userId) * 0.6)].to_csv(
            "E:/data_movie/ml-20m/experiment/trainAll/" + str(i[0]) + "_train.csv", sep=',', index=False)
        i[1].sort_values('timestamp').iloc[round(len(i[1].userId)*0.6):].to_csv("E:/data_movie/ml-20m/experiment/testAll/" + str(i[0]) + "_test.csv", sep=',',index=False)

def get_users():
    names = os.listdir('E:/data_movie/ml-20m/experiment/trainAll')
    f = open("../data/users.txt", 'w', encoding='utf8')
    users = []
    for i in names:
        path = 'E:/data_movie/ml-20m/experiment/trainAll/' + i
        userId = i.replace('_train.csv', '')
        #print(userId)
        users.append(int(userId))
    users.sort()
    for item in users:
        f.write(str(item) + '\n')

def get_train_pair():
    user2id = {}
    with open('../data/users.txt') as f:
        index = 0
        for line in f:
            user2id[line.strip()] = index
            index += 1
    item2id = {}
    with open('../data/items.txt') as f:
        index = 0
        for line in f:
            item2id[line.strip()] = index
            index += 1
    print(user2id)
    print(item2id)

    names = os.listdir('E:/data_movie/ml-20m/experiment/trainAll')
    f = open("../data/train_pairs.txt", 'w', encoding='utf8')
    f1 = open("../data/valid_pairs.txt", 'w', encoding='utf8')
    train_pair = []
    valid_pair = []
    for i in names:
        all_pair = []
        path = 'E:/data_movie/ml-20m/experiment/trainAll/' + i
        userId = i.replace('_train.csv', '')
        print(userId)
        movies = pd.read_csv(path, header=0)
        for index, row in movies.iterrows():
            # pair = str(user2id[str(userId)]) + ',' + str(item2id[str(int(row['movieId']))]) + ',' + str(
            #     str(int(row['timestamp'])))
            pair = str(user2id[str(userId)]) + ',' + str(item2id[str(int(row['movieId']))])
            all_pair.append(pair)

        train_pair.extend(all_pair[:-1])
        valid_pair.append(all_pair[-1])
    for item in train_pair:
        f.write(item + '\n')
    for item in valid_pair:
        f1.write(item + '\n')

def get_test_pair():
    user2id = {}
    with open('../data/users.txt') as f:
        index = 0
        for line in f:
            user2id[line.strip()] = index
            index += 1
    item2id = {}
    with open('../data/items.txt') as f:
        index = 0
        for line in f:
            item2id[line.strip()] = index
            index += 1
    print(user2id)
    print(item2id)

    names = os.listdir('E:/data_movie/ml-20m/experiment/testAll')
    f = open("../data/test_pairs.txt", 'w', encoding='utf8')
    all_pair = []
    for i in names:

        path = 'E:/data_movie/ml-20m/experiment/testAll/' + i
        userId = i.replace('_test.csv', '')
        print(userId)
        movies = pd.read_csv(path, header=0)
        for index, row in movies.iterrows():
            # pair = str(user2id[str(userId)]) + ',' + str(item2id[str(int(row['movieId']))])+ ',' + str(str(int(row['timestamp'])))
            pair = str(user2id[str(userId)]) + ',' + str(item2id[str(int(row['movieId']))])
            all_pair.append(pair)

    for item in all_pair:
        f.write(item + '\n')

def get_train_csv():
    data=[]
    with open('../data/train_pairs.txt') as f:
        for line in f:
            user=int(line.strip().split(',')[0])
            item=line.strip().split(',')[1]
            time=line.strip().split(',')[2]
            data_1=[]
            data_1.append(user)
            data_1.append(item)
            data_1.append(time)
            data.append(data_1)
    df = pd.DataFrame(np.array(data),columns=['user', 'item', 'time'])
    print(df)
    #df.sort_values('user')

    df['user']=df['user'].astype('int')
    print(df.sort_values('user'))
    df.sort_values('user').to_csv('../data/train.csv', sep=',', index=False)

def get_pos_neg_user_aspect():
    df = pd.read_csv('../data/result.csv', header=0)
    f1=open('../data/user_pos_aspect_rank.txt', 'w', encoding='utf8')
    f2 = open('../data/user_neg_aspect_rank.txt', 'w', encoding='utf8')

    movieId = df['0']
    movie_feature = df.drop('0', 1)
    movie_feature_normalized = np.insert(preprocessing.normalize(movie_feature, norm='l2'), 0, values=movieId, axis=1)
    all_tags_mean = movie_feature.mean().tolist()
    path = '../data/train.csv'
    train = pd.read_csv(path, header=0)
    groups = train.groupby(by='user')
    user_interest = []
    userIds = []
    for group in groups:
        # print(group[1])
        userId = group[1].user.tolist()[0]
        userIds.append(userId)
        items = group[1].item
        df2 = df.loc[df['0'].isin(items)].drop('0', 1)
        mean = df2.mean()
        interest = mean - all_tags_mean  # preference

        user_interest.append(interest)
        #print(interest)
        aspects = []
        x = np.array(interest)
        aspect2interest={}
        index=1
        for i in interest:
            aspect2interest[index]=i
            index+=1
        print(aspect2interest)

        pos_aspects={}
        neg_aspects={}
        for key in aspect2interest.keys():

            if aspect2interest[key]>0:
                pos_aspects[key]=aspect2interest[key]
            else:
                neg_aspects[key]=aspect2interest[key]
        pos_aspects_rank=sorted(pos_aspects.items(),key=lambda x:x[1],reverse=True)
        neg_aspects_rank=sorted(neg_aspects.items(),key=lambda x:x[1],reverse=False)

        f1.write(','.join([str(x[0]) for x in pos_aspects_rank])+ '\n')
        f2.write(','.join([str(x[0]) for x in neg_aspects_rank])+ '\n')

def get_user_aspect_baseline():#noneg
    df = pd.read_csv(
        '../data/result.csv',
        header=0)
    f2 = open('../data/user_aspect_rank_baseline.txt', 'w', encoding='utf8')
    path = '../data/train.csv'
    train = pd.read_csv(path, header=0)
    groups = train.groupby(by='user')
    for group in groups:
        # print(group[1])
        userId = group[1].user.tolist()[0]
        print(userId)
        items = group[1].item
        #print(items)
        df2 = df.loc[df['0'].isin(items)].drop('0', 1)
        sum=df2.sum().tolist()
        result=[]
        #print(sum)
        index=0
        for i in sum:
            index=index+1
            if(i>2):
                inner=[index,i]
                result.append(inner)
        if(len(result)==0):
            f2.write('\n')
            continue
        data=np.array(result)
        data=data[np.lexsort(-data.T)][:,0].tolist()
        #print(result)
        #print(data)
        f2.write(','.join([str(x) for x in data]))
        f2.write('\n')
        #break


#get_user_movieId()
#get_users()
get_train_pair()
get_test_pair()
#get_train_csv()
#get_pos_neg_user_aspect()
# get_user_aspect_baseline()
