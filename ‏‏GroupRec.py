import pandas as pa
import math
import random
import numpy as np
import datetime


random.seed(1)

#class for group recommended and profile generated
class GroupRec():

    def __init__(self):
        self.train = []
        self.test = []
        self.groups = {}
        self.profiles = {}
        # self.trainItems = []
        # self.trainUsers = []
        self.trainDict = {}
        self.testDict = {}
        self.train_records = []
        self.test_records = []
        self.testItems = []

#load data from csv
    def loadRating(self):
        data = pa.read_csv("ratings.csv")
        data.drop(data.columns[[3]], axis=1, inplace=True)
        self.train = data.sample(frac=0.5, random_state=200)
        self.train_records = self.train.to_records(index=False)
        self.test = data.drop(self.train.index)
        self.test_records = self.test.to_records(index=False)
        for x in self.train_records:
            user = x[0]
            item = x[1]
            rating = x[2]
            if user in self.trainDict:
                self.trainDict[user][item] = rating
            else:
                self.trainDict[user] = {}
                self.trainDict[user][item] = rating
        for x in self.test_records:
            user = x[0]
            item = x[1]
            rating = x[2]
            if item in self.testItems:
                continue
            else:
                self.testItems.append(item)
            if user in self.testDict:
                self.testDict[user][item] = rating
            else:
                self.testDict[user] = {}
                self.testDict[user][item] = rating

#create random groups
    def createGroups(self, size, groupsNum):
        grPercent = size / len(self.train)
        for i in range(0, groupsNum):
            self.groups[i] = self.train.sample(frac=grPercent, replace=True, random_state=i).to_records(index=False)
        return self.groups

#check if user have rating on item 1 and 2
    def checkIfExists(self, user, ratings, item1, item2):
        if item1 in ratings[user] and item2 in ratings[user]:
            return 1
        return 0

#calculate MCS group generator
    def calcMCS(self, groupUsers, ratings, item1, item2):
        groupMCS = {}
        keysList = []
        for u in groupUsers:
            res = self.checkIfExists(u, ratings, item1, item2)
            if res == 1:
                groupMCS[u] = 1
        if len(groupMCS) == 0:
            return -1
        # TODO: to improve
        for k, v in groupMCS.items():
            keysList.append(k)
        random_index = random.randrange(0, len(keysList))
        return keysList[random_index]

#calculate user weights in group profile
    def calcUserWeightsInProfileGroup(self, group,groupTrain=[],usersDist={},improve=False):
        MCS_total = 0
        MCS_allUsers = {}
        groupItems = []
        ratings = {}
        for u in group['userId']:
            MCS_allUsers[u] = 0
        for u in self.train_records:
            user = u[0]
            item = u[1]
            rate = u[2]
            if user in group['userId']:
                if user in ratings:
                    ratings[user][item] = rate
                else:
                    ratings[user] = {}
                    ratings[user][item] = rate
                if item in groupItems:
                    continue
                groupItems.append(item)
        weights = {}
        groupProfile = {}
        totalItems = len(groupItems)
        groupItems.sort()
        for i in groupItems:
            for j in groupItems:
                if i >= j:
                    continue
                if improve:
                    userMCS = self.calcMCSImprove(group['userId'], groupTrain ,usersDist ,i, j)
                else:
                    userMCS = self.calcMCS(group['userId'], ratings, i, j)
                if userMCS != -1:
                    MCS_allUsers[userMCS] += 1
        for u in MCS_allUsers:
            MCS_allUsers[u] = MCS_allUsers[u] * 2 / totalItems
            MCS_total += MCS_allUsers[u]
        for u in MCS_allUsers:
            if MCS_total == 0:
                weights[u] = 0
            else:
                weights[u] = MCS_allUsers[u] / MCS_total
#        print("weights")
#        print(weights)
        for userId in ratings:
            for itemId in ratings[userId]:
                if itemId in groupProfile:
                    groupProfile[itemId] += weights[userId] * ratings[userId][itemId]
                else:
                    groupProfile[itemId] = weights[userId] * ratings[userId][itemId]
        return groupProfile, groupItems

#calculate average for simmilarity
    def avg(self, g):
        sum = 0
        count = len(g)
        for i in g:
            sum += g[i]
        avg = sum / count
        return avg

#calculate simmilarity
    def sim(self, g, u, g_avg):
        u_avg = self.avg(u)
        mone = 0
        sum_g = 0
        sum_u = 0
        for item in g:
            if item in u:
                rg = g[item] - g_avg
                ru = u[item] - u_avg
                mone += rg * ru
                sum_g += rg * rg
                sum_u += ru * ru
        mechane = math.sqrt(sum_g * sum_u)
        if mechane == 0:
            res = 0
        else:
            res = mone / mechane
        return res, u_avg
    
#calculate neighbors average     
    def calcNeighborsItemAvg(self, item, neighborsId):
        sum = 0
        count = 0
        for neighbor in neighborsId:
            if item in self.trainDict[neighbor]:
                sum += self.trainDict[neighbor][item]
                count += 1.0
        if count == 0:
            return 0
        else:
            return sum/count
    
#find neighbors of the group
    def findNeighbors(self, g, gUsers):
        totalSimBetweenNeighbors = 0
        g_avg = self.avg(g)
        sim_neighbors = {}
        neighbor_param = 0
        for user in self.trainDict:
            if user in gUsers:
                continue
            similarity, u_avg = self.sim(g, self.trainDict[user], g_avg)
            if similarity > neighbor_param:
                sim_neighbors[user] = similarity
                totalSimBetweenNeighbors += abs(similarity)
#        print("sim:")
#        print(sim_neighbors)
#        print("neighbors num:")
#        print(len(sim_neighbors))
        return sim_neighbors, totalSimBetweenNeighbors

#normalize rating
    def calcDelta_Rp(self, delta_r):
        r_min = 0
        r_max = 5
        diff = r_max - r_min
        if delta_r >= 0 and delta_r < diff / 3:
            return 1
        if delta_r >= diff / 3 and delta_r < 2 * diff / 3:
            return 0.5
        else:
            return 0

#calculate relevvance items for predictions
    def calcRelevanceOfItem(self, item, itemToPredict, neighborsId, groupProfile):
        total = 0
        k = 0
        for neighbor in neighborsId:
            if item in self.trainDict[neighbor] and itemToPredict in self.trainDict[neighbor]:
                #print("checkkkkkkkkkkk")
                dist = abs(self.trainDict[neighbor][item] - self.trainDict[neighbor][itemToPredict])
                delta_Rp = self.calcDelta_Rp(dist)
                total += delta_Rp
                k += 1.0
        if k == 0:
            return 0.0
        else:
            return total / k

#check if item relevant
    def checkIfItemIsRelevant(self, threshold, item, itemToPredict, neighborsId, groupProfile):
        Rel = self.calcRelevanceOfItem(item, itemToPredict, neighborsId, groupProfile)

        if Rel >= threshold:
            return 1
        else:
            return 0

#calculate average for item that simmilar to traget item
    def calcLocalAverageRating(self, threshold, itemToPredict, neighborsId, groupProfile):
        sum = 0
        count = 0
        for item in groupProfile:
            #print("item")
            #print(item)
            if item == itemToPredict:
                continue
            if self.checkIfItemIsRelevant(threshold, item, itemToPredict, neighborsId, groupProfile) == 1:
                #print(item)
                sum += self.calcNeighborsItemAvg(item, self.trainDict)
                count += 1.0
        if count == 0:
            return 0
        else:
            return sum / count

#predict item rating
    def predictItemRating(self, itemToPredict, groupProfile, simNeighbors, totalSimBetweenNeighbors):
        diff = 0
        keys=random.sample(groupProfile.keys(),200)
        neighborsIds = []
        totalSim = 0
        for neighbor in simNeighbors:
            neighborsIds.append(neighbor)
            if itemToPredict in self.trainDict[neighbor]:
                sim = simNeighbors[neighbor]
                totalSim += sim
                diff += (self.trainDict[neighbor][itemToPredict] - self.avg(self.trainDict[neighbor])) * sim
            # else:
            # diff += self.avg(self.trainDict[neighbor])*simNeighbors[neighbor]
        # diff /= totalSimBetweenNeighbors
        if totalSim == 0:
            totalSim = 1
        diff /= totalSim

        threshold = 0.3  # TODO
        #R_g_i_avg = self.checkIfItemIsRelevant(threshold, itemToPredict, neighborsIds, groupProfile)
        R_g_i_avg = self.calcLocalAverageRating(threshold, itemToPredict, neighborsIds, keys)
        #print("itemToPredict")
        #print(itemToPredict)
        #print("R_g_i_avg")
        #print(R_g_i_avg)
        R_g_i = R_g_i_avg + diff
        return R_g_i

#predict test ratings
    def predictAllRatings(self, groupProfile, simNeighbors, totalSimBetweenNeighbors, groupItems):
        predictions = {}
        for item in self.testItems:
            if item in groupItems:
                continue
            predictions[item] = self.predictItemRating(item, groupProfile, simNeighbors, totalSimBetweenNeighbors)
        return predictions

#calculate baselines least misery, average, average with least misery
    def BaseLines(self, group, AMminVal):
        df = pa.DataFrame(group).drop(['movieId', 'rating'], axis=1)
        groupdata = pa.merge(self.train, df, left_on='userId', right_on="userId", how='right')
        groupDataAM = groupdata[(groupdata.rating > AMminVal)]

        #        print(groupdata)
        f = {'rating': ['min', 'mean']}
        base = groupdata.groupby(['movieId']).agg(f)
        base.columns = ["_".join(x) for x in base.columns.ravel()]
        LM = base.drop(['rating_mean'], axis=1)
        AVG = base.drop(['rating_min'], axis=1)
        AM = groupDataAM.groupby(['movieId']).agg({'rating': 'mean'}).rename(columns={'rating': 'rating_AM'})
        return LM, AVG, AM

 
#calculate predictions for baselines with user base CF
    def predictionBaseline(self,dfProfile,groupTrain,groupTest,gUsers):
        recommended={}
        avgProfile=dfProfile.mean().values[0]
        sim_neighbors, totalSimBetweenNeighbors = self.findNeighbors(pandaToDict(dfProfile), gUsers)
        sim_neighborsDF=pa.DataFrame.from_dict(sim_neighbors,orient='index').reset_index().rename( columns={'index': 'userId',0 : 'sim'})
        groupTestItems = groupTest.drop(['userId','rating'],axis=1).to_records()
        trainWithoutGroup=self.train[~self.train['userId'].isin(gUsers)]
        trainWithoutGroup=pa.merge(trainWithoutGroup,sim_neighborsDF,left_on=['userId'], right_on=['userId'],how='left').dropna()
        f = {'rating': ['mean']}
        usersAVG=trainWithoutGroup.groupby('userId').agg(f).reset_index().rename(columns={'rating': 'ratingAvg'})
        usersAVG.columns = usersAVG.columns.droplevel(1)
        trainWithoutGroup=pa.merge(trainWithoutGroup,usersAVG,left_on=['userId'], right_on=['userId'],how='inner')
        trainWithoutGroup['ratingMinusAvg']=trainWithoutGroup['rating']-trainWithoutGroup['ratingAvg']
        trainWithoutGroup['numerator']=trainWithoutGroup['ratingMinusAvg']*trainWithoutGroup['sim']
#        print(trainWithoutGroup)

        fnew={'numerator' : ['sum'], 'sim' : ['sum']}
        for index,movie in groupTestItems:
            Numerator,denominator=trainWithoutGroup[trainWithoutGroup.movieId==movie].agg(fnew).values[0]
            if Numerator==0 or denominator==0:
                recommended[movie]=0
            else:
                recommended[movie]=avgProfile+ Numerator/denominator
        
        return recommended
    
#NDCG measure for ranking the list of recommended    
    def NDCG(self,group,groupTrain,groupTest, recommended):

        recommendedDF=pa.DataFrame.from_dict(recommended,orient='index').reset_index().rename( columns={'index': 'movieIdRec',0 : 'ratingRec'})#.sort_values(by=['ratingRec'], ascending=False)
#        print(recommendedDF)
        index=1
        ndcg=0
        for u,i,r in group:
            dcg=0
            idcg=0
            groupTestUser=groupTest[groupTest.userId==u]
            dcg=DCGandIDCG(groupTestUser,recommendedDF,True)
            idcg=DCGandIDCG(groupTestUser,recommendedDF,False)
            if dcg==-1 or idcg==-1:
                continue
            ndcg=ndcg+dcg/idcg
            index+=1
        
        return ndcg/index
    
#Fmeasure for measure ratings predictions quality
    def Fmeasure(self,group,groupTrain,groupTest, recommended,Fthreshold):
             
        recommendedDF=pa.DataFrame.from_dict(recommended,orient='index').reset_index().rename( columns={'index': 'movieIdRec',0 : 'ratingRec'})
        df = pa.merge(groupTest,recommendedDF,left_on=['movieId'], right_on=['movieIdRec'],how='inner')
        df = df.drop(['userId','movieIdRec'], axis=1)
        df['TP']=df.apply(lambda row: 1 if row['rating'] >= Fthreshold and row['ratingRec'] >= Fthreshold else 0,axis=1)
        df['FN']=df.apply(lambda row: 1 if row['rating'] >= Fthreshold and row['ratingRec'] < Fthreshold else 0,axis=1)
        df['FP']=df.apply(lambda row: 1 if row['rating'] < Fthreshold and row['ratingRec'] >= Fthreshold else 0 ,axis=1)
        f={'movieId': ['count'], 'TP': ['sum'],'FN': ['sum'],'FP': ['sum']}
#        print(df)
        dfnew=df.groupby('movieId').agg(f)
        dfnew.columns = ["_".join(x) for x in dfnew.columns.ravel()]
        dfnew=dfnew.reset_index()
#        print(dfnew)
        dfnew['fmovie']=dfnew.apply(lambda row: Fmovie(row), axis=1)
#        print(dfnew)
        dfF=dfnew.groupby(['fmovie'])['fmovie'].count()
        fDict=dfF.to_dict()
        if 'TP' not in fDict:
            fDict['TP']=0
        if 'FP' not in fDict:
            fDict['FP']=0
        if 'FN' not in fDict:
            fDict['FN']=0
            
            
        precision = fDict['TP'] / (fDict['TP'] + fDict['FP'])
        recall=fDict['TP']/(fDict['TP']+fDict['FN'])
        F=2*(precision*recall/(precision+recall))
        if math.isnan(F):
            return 0
        return F
        
#calc improve MCS with user rating distribution and avg in pairs
    def calcMCSImprove(self, gUsers,groupTrain,usersDist, item1, item2):
        keysList=[]
        groupSim={}
        ratings={}
        np.random.seed(0)
        
        if(random.random()<0.85):
            return -1
#        print("item 1 = " + str(item1) + " item 2 = "+str(item2))
        groupTrain=groupTrain[(groupTrain.movieId==item1)  | (groupTrain.movieId==item2)].to_records(index=False)
        item1sum=0
        item2sum=0
        for u,m,r in groupTrain:
            if u not in ratings:
                ratings[u] = {}
            ratings[u][m] = r
            if item1==m:
                item1sum=item1sum+r
            if item2==m:
                item2sum=item2sum+r
            
        for u in gUsers:
            if u not in ratings:
                ratings[u]={}
            try:
                r=ratings[u][item1]
            except KeyError:
                r=np.random.choice(np.arange(0, 5,0.5), p=usersDist[u])
                ratings[u][item1]=r
                item1sum=item1sum+r
            try:
                r=ratings[u][item2]
            except KeyError:
                r=np.random.choice(np.arange(0, 5,0.5), p=usersDist[u])       
                ratings[u][item2]=r
                item2sum=item2sum+r
        ratings['AVG']={}
        ratings['AVG'][item1]=item1sum/len(gUsers)
        ratings['AVG'][item2]=item2sum/len(gUsers)        
        pair_avg=(ratings['AVG'][item1]+ratings['AVG'][item2])/2
        
        maxSim=0
        for u in ratings:
            if u=='AVG':
                continue
            simUser,userAvg =  self.sim(ratings['AVG'], ratings[u], pair_avg)
            groupSim[u]=simUser
            if simUser>maxSim:
                maxSim=simUser
        
        
        for k, v in groupSim.items():
            if v==maxSim:
                keysList.append(k)
        random_index = random.randrange(0, len(keysList))
        return keysList[random_index]
        
#calculate user rating distribution
def userDistribution(data,gUser):
    usersProb={}
    dist=np.zeros((10,), dtype=int)
    for u in gUser:
        datauser=data[(data.userId==u)]
        arr=np.array(datauser['rating'])
        for x in arr:
            dist[int(x*2)-1]=dist[int(x*2)-1]+1
        s=np.sum(dist)    
        puser=dist/s
        usersProb[u]=puser
        dist=np.zeros((10,), dtype=int)

    return (usersProb)


#fmeasure check user TP,FP,FN
def Fmovie(row):
    if row['FN_sum'] > 0:
        return 'FN'
    elif row['FP_sum'] == row['movieId_count']:
        return 'FP'
    elif row['TP_sum'] == row['movieId_count']:
        return 'TP'


#calculate DCG and IDCG for group of users
def DCGandIDCG(groupTestUser, recommendedDF, dcg):
    df = pa.merge(groupTestUser, recommendedDF, left_on=['movieId'], right_on=['movieIdRec'], how='inner')
    if df.empty:
        return -1
    
    if (dcg):
        df = df.sort_values(by=['ratingRec'], ascending=False)
    else:
        df = df.sort_values(by=['rating'], ascending=False)
    df = df.reset_index()
    df['pos'] = df.index
    #    print(df)
    df['dcg_item'] = df.apply(lambda row: row['rating'] if row['pos'] + 1 == 1 else row['rating'] / (math.log2(row['pos'] + 1)), axis=1)
    #    print(df)
    return df['dcg_item'].sum()


#arrange Test Data by removing unnecessary rows
def arrangeTestData(test,train,group):
    df=pa.DataFrame(group)
    if 'movieId' in df.columns:
        df = df.drop(['movieId', 'rating'], axis=1)
    else:
        df=df.rename(columns={0 : 'userId'})
    groupTrain = pa.merge(train, df, left_on='userId', right_on="userId", how='right')
    groupTest = pa.merge(test, df, left_on='userId', right_on="userId", how='right')
    combine = groupTest.merge(groupTrain,left_on=['userId','movieId'], right_on=['userId','movieId'],how='inner')

#    groupTest=groupTest[~groupTest.isin(combine)]
    groupTest=groupTest.drop(combine.index) 
#    print(groupTest)         
    return groupTrain,groupTest  

#convert data structure from panda to dictionary
def pandaToDict(df):
    df.columns = range(df.shape[1])
    df_dict=df.to_dict()
    if 0 in df_dict:
        return df_dict[0]
    return df_dict

#write to file profile evaluation by measures result
def writeToFile(new, text,groupSize):
    if new == 0:
        file = open("measures"+str(groupSize)+".txt", "w")
        file.write('group,NDCG: profile, LM, AVG, AVGLM, Fmeasure: profile, LM, AVG, AVGLM,profileIMP_NDCG,  profileIMP_Fmeasure, ' + str(datetime.datetime.now()) + "\n")
    if new == 1:
        file = open("measures"+str(groupSize)+".txt", "a")
        file.write(text + ',')
    if new == 2:
        file = open("measures"+str(groupSize)+".txt", "a")
        file.write(str(datetime.datetime.now()) + "\n")
    file.flush()
    file.close()

def main():
    gr = GroupRec()
    gr.loadRating()
    LM, AVG, AM = pa.DataFrame(), pa.DataFrame(), pa.DataFrame()
    groupSize = 5
    groupsNum = 10
    AMminVal = 2
    NDCG_newProfiles=[]
    NDCG_ProfilesLM=[]
    NDCG_ProfilesAVG=[]
    NDCG_ProfilesAM=[]
    NDCG_newProfilesIMP=[]
    Fmeasure_newProfiles=[]
    Fmeasure_ProfilesLM=[]
    Fmeasure_ProfilesAVG=[]
    Fmeasure_ProfilesAM=[]
    Fmeasure_newProfilesIMP=[]
    T=3
    
    writeToFile(0,'',str(groupSize))
    groups = gr.createGroups(groupSize, groupsNum)
    length = len(groups)
    for i in range(0, length):
        print("Group: "+ str(i))
#        print(groups[i]['userId'])
        
        writeToFile(1, str(i),str(groupSize))
        groupProfile, groupItems = gr.calcUserWeightsInProfileGroup(groups[i])
#        print("groupProfile")
#        print(groupProfile)
        sim_neighbors, totalSimBetweenNeighbors = gr.findNeighbors(groupProfile, groups[i]['userId'])
        predictions = gr.predictAllRatings(groupProfile, sim_neighbors, totalSimBetweenNeighbors, groupItems)
#        print("predictions")
#        print(predictions)
        LM, AVG, AM = gr.BaseLines(groups[i], AMminVal)
        
        groupTrain,groupTest = arrangeTestData(gr.test,gr.train,groups[i]['userId'])
        predictionsLM= gr.predictionBaseline(LM,groupTrain,groupTest, groups[i]['userId'])
        predictionsAVG= gr.predictionBaseline(AVG,groupTrain,groupTest, groups[i]['userId'])
        predictionsAM= gr.predictionBaseline(AM, groupTrain,groupTest, groups[i]['userId'])
        
        print('***********NDCG*************')
        print("NDCG new profile:")
        NDCG_newProfiles.append(gr.NDCG(groups[i],groupTrain,groupTest,predictions))
        print(NDCG_newProfiles)
        writeToFile(1, str(NDCG_newProfiles[i]),str(groupSize))
        
        print("\nNDCG Baseline predictions:")
        print("Least misery:")
        NDCG_ProfilesLM.append(gr.NDCG(groups[i],groupTrain,groupTest,predictionsLM))
        print(NDCG_ProfilesLM)
        writeToFile(1, str(NDCG_ProfilesLM[i]),str(groupSize))
        
        print("Average:")
        NDCG_ProfilesAVG.append(gr.NDCG(groups[i],groupTrain,groupTest,predictionsAVG))
        print(NDCG_ProfilesAVG)
        writeToFile(1, str(NDCG_ProfilesAVG[i]),str(groupSize))
        
        
        print("Average without misery")
        NDCG_ProfilesAM.append(gr.NDCG(groups[i],groupTrain,groupTest,predictionsAM))
        print(NDCG_ProfilesAM)
        writeToFile(1, str(NDCG_ProfilesAM[i]),str(groupSize))
        
        
        print('\n\n***********F-measure************')
        print("Fmeasure new profile:")
        Fmeasure_newProfiles.append(gr.Fmeasure(groups[i],groupTrain,groupTest,predictions,T))
        print(Fmeasure_newProfiles)
        writeToFile(1, str(Fmeasure_newProfiles[i]),str(groupSize))
        
        print("\nBaseline predictions:")
        print("Least misery:")
        Fmeasure_ProfilesLM.append(gr.Fmeasure(groups[i],groupTrain,groupTest,predictionsLM,T))
        print(Fmeasure_ProfilesLM)
        writeToFile(1, str(Fmeasure_ProfilesLM[i]),str(groupSize))
        
        
        print("Average:")
        Fmeasure_ProfilesAVG.append(gr.Fmeasure(groups[i],groupTrain,groupTest,predictionsAVG,T))
        print(Fmeasure_ProfilesAVG)
        writeToFile(1, str(Fmeasure_ProfilesAVG[i]),str(groupSize))
        

        print("Average without misery")
        Fmeasure_ProfilesAM.append(gr.Fmeasure(groups[i],groupTrain,groupTest,predictionsAM,T))
        print(Fmeasure_ProfilesAM)
        writeToFile(1, str(Fmeasure_ProfilesAM[i]),str(groupSize))


        usersDist = userDistribution(groupTrain,groups[i]['userId'])
        groupProfileIMP,groupItems = gr.calcUserWeightsInProfileGroup(groups[i],groupTrain,usersDist,True)
        print('****************************')
#        print(groupProfileIMP)
        
        sim_neighborsIMP, totalSimBetweenNeighborsIMP = gr.findNeighbors(groupProfileIMP, groups[i]['userId'])
        predictionsIMP = gr.predictAllRatings(groupProfileIMP, sim_neighborsIMP, totalSimBetweenNeighborsIMP, groupItems)
        
        print('\n\n***********NDCG*************')
        print("NDCG new profile IMPROVE:")
        NDCG_newProfilesIMP.append(gr.NDCG(groups[i],groupTrain,groupTest,predictionsIMP))
        print(NDCG_newProfilesIMP)
        writeToFile(1, str(NDCG_newProfilesIMP[i]),str(groupSize))

        print('\n\n***********F-measure************')
        print("Fmeasure new profile IMPROVE:")
        Fmeasure_newProfilesIMP.append(gr.Fmeasure(groups[i],groupTrain,groupTest,predictionsIMP,T))
        print(Fmeasure_newProfilesIMP)
        writeToFile(1, str(Fmeasure_newProfilesIMP[i]),str(groupSize))
        writeToFile(2, '',str(groupSize))
        
        
        
        


if __name__ == '__main__':
    main()
