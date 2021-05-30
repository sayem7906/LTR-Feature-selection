import numpy as np 
from sklearn.preprocessing import normalize
import copy
from bitarray import bitarray
import xgboost as xgb
from Metrics.metrics import ndcg_mean,mean_ap


class  simulated_annealing:
    def __init__(self,model_param, n_steps, k=1,
                 temperature=.65,num_acc_steps=5,
                 num_not_best_step=15,options = None):
        
        self.k = k
        """number of features to be taken"""
        
        self.model_param = model_param
        """the model to evaluate with"""
        
        self.n_steps = n_steps
        """number of iterations"""
        
        self.init_temperature = temperature
        # self.cooldown_factor = cooldown_factor
        
        self.num_acc_steps = num_acc_steps
        
        '''progress parameter'''
        # self.num_not_best_step = num_not_best_step
        # self.num_not_best_step = -1
        
        self.options = options
        
        
        # self.metric = metric
        
        self.best_features = None
        """Best set(bitarray) of features found."""


        self.best_score = dict()
        """test score when evaluating best set of features found. """


    def fit(self,x_train,x_valid,x_test,y_train,y_valid,y_test,group_train,
            group_valid,group_test):
        
        
        feature_no = x_train.shape[1]
        self.feature_set = set(np.arange(feature_no))
        visited = dict()
        """dictionary that keeps the states and their scores that were already visited"""
        state,feature_subset = self.init_state(feature_no,self.k)
        
        temperature = self.init_temperature

        best_state = None
        best_pred = None
        best_score = -np.inf
        score = -np.inf
        cur_state = copy.deepcopy(state)
        cur_feature_subset = copy.deepcopy(feature_subset)
        
        # print('init temp: ',temperature)
        """used for monitoring current number of updated states"""
        ct_acc_step = 0
        
        """used for checking the number of times loop went without accepting"""
        # ct_not_best_step = 0
        
        for i in range(self.n_steps):
            
            # print("cur step: ",i,"===============\n===============\n===============\n")
            # print("cur state: ",cur_state)
            
            '''progress parameter checker update'''
            # ct_not_best_step += 1
            
            if(cur_state.tobytes() not in visited):
                """update train and test data with selected features"""
                temp = np.where(np.array(list(cur_state))==True)[0]
                x_train_temp = x_train[:,temp]
                
                x_test_temp = x_test[:,temp]
                
                x_valid_temp = x_valid[:,temp]
                # print(x_train_temp[0:5,:])
                
                
                """train model and generate new score"""
                model = xgb.sklearn.XGBRanker(**self.model_param)
                
                model.fit(x_train_temp,y_train,group=group_train,verbose=False,
                          eval_set=[(x_valid_temp, y_valid)], eval_group=[group_valid])
                predictions = model.predict(x_test_temp)
                new_score = ndcg_mean(group_test, y_test, predictions, 10)
                # print('new score ',new_score)
                
                visited[cur_state.tobytes()] = new_score
                
            else:
                # print('ohnooo--',cur_state)
                
                new_score = visited[cur_state.tobytes()]
            # print('current score: ',new_score,'\n\n')
            if new_score<score:
                accept = self.metropolice(new_score,score,temperature)
                # accept = -1
            else:
                accept = 1
            if np.random.uniform(0,1) <= accept:
                
    
                ct_acc_step +=1
                
                state = copy.deepcopy(cur_state)
                score = new_score
                feature_subset = copy.deepcopy(cur_feature_subset)
                
                if(score>best_score):
                    '''progress parameter checker update'''
                    # ct_not_best_step = 0
                    
                    best_state = copy.deepcopy(state)
                    best_score = score
                    best_pred = predictions
                    # print('ki jhamelaa')
                    
                    # best_feature_subset = copy.deepcopy(feature_subset)
                    # print(best_state)
                    # print('\n\n')
                    # input('enter: ')
                """updating temperature after a number of accepted states"""
                if(ct_acc_step ==self.num_acc_steps):
                    temperature = self.cooldown(i+1,temperature)
                    ct_acc_step = 0
                    # print(temperature)
            
            """after not updating best state for a time restart from previous best state"""
            # if(ct_not_best_step == self.num_not_best_step):
            #     ct_not_best_step = 0
            #     score = best_score
            #     state,feature_subset = copy.deepcopy(best_state),copy.deepcopy(best_feature_subset)
                
            cur_state,cur_feature_subset = self.select_neighbor(copy.deepcopy(state),copy.deepcopy(feature_subset))
            # cur_state = self.select_neighbor_inverse(copy.deepcopy(state))
            # cur_state = self.select_neighbor_insert(copy.deepcopy(state))
        self.best_score['ndcg'] =  best_score
        # print(best_pred)
        self.best_score['map'] = mean_ap(group_test, y_test, best_pred)
        self.best_features = best_state
        
        return self
                
                
            
    def init_state(self,n,k):
        
        a =bitarray(n)
        l = set()
        a.setall(False)
        rand_ids = np.random.permutation(n)
        for i in range(k):
            l.add(rand_ids[i])
            a.invert(rand_ids[i])
        return a,l
    
    
    
    def metropolice(self,new_score,score,temperature):
        return np.exp((new_score-score)/ temperature)
    
    
    
    
    def cooldown(self,step,current_temperature):
       if self.options['schedule']==1:
           return current_temperature*self.options['cooldown_factor']
       elif self.options['schedule']==2:
           return self.init_temperature/np.log(10+step)
       return self.init_temperature/(1+step)
    
    
    
    
    
    def select_neighbor(self,state,subset):
        
        
        if self.options['neighbor']==1:
            """swap based neighbor selection"""
            
            
            candidates = list(self.feature_set - subset)
            
            
            temp_subset = list(subset)
            new_state = copy.deepcopy(state)
            
            
            """choosing random feature pair to swap"""
            subset_id = np.random.randint(len(temp_subset))
            candidate_id = np.random.randint(len(candidates))
            
            """updating state"""
            new_state.invert(temp_subset[subset_id])
            new_state.invert(candidates[candidate_id])
            
            
            """updating feature subset"""
            subset.remove(temp_subset[subset_id])
            subset.add(candidates[candidate_id])
            
            return new_state,subset
        
        
        else:
            """insertion based neighbor selection"""
            # candidates = list(self.feature_set - subset)
        
        
            # temp_subset = list(subset)
            new_state = copy.deepcopy(state)
            
            j=k=0
            """choosing random feature pair to swap"""
            while j==k:  
                j = np.random.randint(len(new_state))
                k = np.random.randint(len(new_state))
            
            
            
            """updating state"""
            new_state.insert(k+1,new_state[j])
            if j<k:
                new_state.pop(j)
            else:
                new_state.pop(j+1)
             
            
            return new_state,None
            
            
        
   
   
    
    # def select_neighbor_inverse(self,state):
    #     # candidates = list(self.feature_set - subset)
        
        
    #     # temp_subset = list(subset)
    #     new_state = copy.deepcopy(state)
        
    #     j=k=0
    #     """choosing random feature pair to swap"""
    #     while j==k:  
    #         j = np.random.randint(len(new_state))
    #         k = np.random.randint(len(new_state))
    #     if(k>j):
    #         j,k = k,j
        
        
    #     """updating state"""
    #     for i in range((k-j+1)//2):
    #         temp = new_state[j+i]
    #         # print('first ',new_state[j+i])
    #         new_state[j+i] = new_state[k-i]
    #         # print('second ',new_state[k-i])
    #         new_state[k-i] = temp
    #         # print('======')
         
        
    #     return new_state
    
    
    # def select_neighbor_insert(self,state):
    #     # candidates = list(self.feature_set - subset)
        
        
    #     # temp_subset = list(subset)
    #     new_state = copy.deepcopy(state)
        
    #     j=k=0
    #     """choosing random feature pair to swap"""
    #     while j==k:  
    #         j = np.random.randint(len(new_state))
    #         k = np.random.randint(len(new_state))
        
        
        
    #     """updating state"""
    #     new_state.insert(k+1,new_state[j])
    #     if j<k:
    #         new_state.pop(j)
    #     else:
    #         new_state.pop(j+1)
         
        
    #     return new_state
    
    
    # def select_neighbor_genetic(self,state):
    #     # candidates = list(self.feature_set - subset)
        
    #     random_state,_ = self.init_state(n = len(state), k = self.k)
        
        
    #     # temp_subset = list(subset)
    #     new_state = copy.deepcopy(state)
        
    #     """Crossing of two states"""
    #     crossover_point = len(new_state)//2
    #     new_state = new_state[0:crossover_point]
    #     new_state.extend(random_state[crossover_point:])
        
        
    #     """equiting the number of required features"""
    #     new_num_ones = set(np.where(np.array(list(new_state))==1)[0])
    #     if len(new_num_ones)>self.k:
    #         rand_ids = np.random.permutation(len(new_num_ones))
    #         new_num_ones = list(new_num_ones)
    #         for i in range(len(new_num_ones) - self.k):
    #             idx = rand_ids[i]
    #             new_state.invert(new_num_ones[idx])
    #     elif len(new_num_ones)<self.k:
    #         new_num_zeros = list(self.feature_set - new_num_ones)
    #         rand_ids = np.random.permutation(len(new_num_zeros))
    #         for i in range(self.k -  len(new_num_ones)):
    #             idx = rand_ids[i]
    #             new_state.invert(new_num_zeros[idx])
        
    #     return new_state
       
            
            
            
            
    