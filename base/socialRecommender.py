from base.iterativeRecommender import IterativeRecommender
from data.social import Social
from data.item import Item
from util import config
from os.path import abspath
class SocialRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,relation,itemRelation,fold='[1]'):
        super(SocialRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.social = Social(self.config, relation) #social relations access control
        self.item = Item(self.config,itemRelation)
        # data clean
        cleanList = []
        cleanPair = []
        for user in self.social.followees:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followees[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followees[u]
        for pair in cleanPair:
            if pair[0] in self.social.followees:
                del self.social.followees[pair[0]][pair[1]]
        cleanList = []
        cleanPair = []
        for user in self.social.followers:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followers[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followers[u]
        for pair in cleanPair:
            if pair[0] in self.social.followers:
                del self.social.followers[pair[0]][pair[1]]
        idx = []
        for n,pair in enumerate(self.social.relation):
            if pair[0] not in self.data.user or pair[1] not in self.data.user:
                idx.append(n)
        for item in reversed(idx):
            del self.social.relation[item]

        # ==================item====================
        cleanList_item = []
        cleanPair_item = []
        for item in self.item.followees:
            if item not in self.data.item:
                cleanList_item.append(item)
            for i2 in self.item.followees[item]:
                if i2 not in self.data.item:
                    cleanPair_item.append((item, i2))
        for i in cleanList_item:
            del self.item.followees[i]
        for pair in cleanPair_item:
            if pair[0] in self.item.followees:
                del self.item.followees[pair[0]][pair[1]]
        cleanList_item = []
        cleanPair_item = []
        for item in self.item.followers:
            if item not in self.data.item:
                cleanList_item.append(item)
            for i2 in self.item.followers[item]:
                if i2 not in self.data.item:
                    cleanPair_item.append((item, i2))
        for i in cleanList_item:
            del self.item.followers[i]
        for pair in cleanPair_item:
            if pair[0] in self.item.followers:
                del self.item.followers[pair[0]][pair[1]]
        idx = []
        for n, pair in enumerate(self.item.relation):
            if pair[0] not in self.data.item or pair[1] not in self.data.item:
                idx.append(n)
        for user in reversed(idx):
            del self.item.relation[user]

    def readConfiguration(self):
        super(SocialRecommender, self).readConfiguration()
        regular = config.OptionConf(self.config['reg.lambda'])
        self.regS = float(regular['-s'])

    def printAlgorConfig(self):
        super(SocialRecommender, self).printAlgorConfig()
        print('Social dataset:',abspath(self.config['social']))
        print('Social relation size ','(User count:',len(self.social.user),'Relation count:'+str(len(self.social.relation))+')')
        print('Social Regularization parameter: regS %.3f' % (self.regS))
        print('=' * 80)
        print('Item dataset:', abspath(self.config['item']))
        print('Item relation size ', '(item count:', len(self.item.item),
              'Relation count:' + str(len(self.item.relation)) + ')')
        print('Social Regularization parameter: regS %.3f' % (self.regS))
        print('=' * 80)

