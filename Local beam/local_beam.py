import numpy as np 
from sklearn.preprocessing import normalize
import copy
from bitarray import bitarray
import xgboost as xgb
from Metrics.metrics import ndcg_mean,mean_ap

class  Local_Beam_Search:
    def __init__(self,model_param,beam_width, n_steps, k=1,
                 temperature=.65,num_acc_steps=5,
                 num_not_best_step=15,options = None):
        
        self.k = k
        """number of features to be taken"""
        
        self.model_param = model_param
        """the model to evaluate with"""
        
        self.beam_width = beam_width # The main parameter
        
        
        self.n_steps = n_steps
        """number of iterations"""
        
        self.num_acc_steps = num_acc_steps * self.beam_width
        
        self.num_not_best_step = num_not_best_step * self.beam_width
        
        self.options = options
        
        self.best_features = None
        """Best set(bitarray) of features found."""


        self.best_score = None
        self.best_scores = dict()
        """test score when evaluating best set of features found. """
        
        self.beam_k_features = []
        self.beam_k_scores = []
        self.beam_k_states = []


    def fit(self,x_train,x_valid,x_test,y_train,y_valid,y_test,group_train,
            group_valid,group_test):
        
        
        feature_no = x_train.shape[1]
        self.feature_set = set(np.arange(46))#feature_no))
        visited = dict()
        
        """dictionary that keeps the states and their scores that were already visited"""
        
        for i in range(self.beam_width):
            state,feature_subset = self.init_state(feature_no,self.k)
            state = bitarray(state)
            temp = np.where(np.array(list(state))==True)[0]
            x_train_temp = x_train[:,temp]
            x_test_temp = x_test[:,temp]
            x_valid_temp = x_valid[:,temp]
            model = xgb.sklearn.XGBRanker(**self.model_param)    
            model.fit(x_train_temp,y_train,group=group_train,verbose=False,
                      eval_set=[(x_valid_temp, y_valid)], eval_group=[group_valid])
                
            predictions = model.predict(x_test_temp,ntree_limit=self.model_param['n_estimators'])
               
            score = ndcg_mean(group_test, y_test, predictions, 10)

            self.beam_k_states.append(state)
            self.beam_k_features.append(feature_subset)
            self.beam_k_scores.append(score)
        
        
        best_state = None
        best_score = -np.inf
        best_scores = dict()
     
        """used for monitoring current number of updated states """
        ct_acc_step = 0
        
        # used for checking the number of times loop went without accepting
        ct_not_best_step = 0
        
        for j in range(self.n_steps):
            self.beam_k_scores, self.beam_k_states, self.beam_k_features = self.sort_based_on_score(
                self.beam_k_scores, self.beam_k_states, self.beam_k_features)
        
            for i in range(self.beam_width):
                
                ct_not_best_step += 1
                crnt_state_list = self.beam_k_states[i]
                cur_state = bitarray(crnt_state_list)
                
                cur_feat = self.beam_k_features[i]
                cur_score = self.beam_k_scores[i]
                
                next_state,next_feature_subset = self.select_neighbor(copy.deepcopy(cur_state),copy.deepcopy(cur_feat))
                next_state = bitarray(next_state)
                
                if(next_state.tobytes() not in visited):
                    
                    # update train and test data with selected features
                    temp = np.where(np.array(list(next_state))==True)[0]
                    x_train_temp = x_train[:,temp]
                
                    x_test_temp = x_test[:,temp]
                
                    x_valid_temp = x_valid[:,temp]
                
                    # train model and generate new score
                    model = xgb.sklearn.XGBRanker(**self.model_param)
                
                    model.fit(x_train_temp,y_train,group=group_train,verbose=False,
                              eval_set=[(x_valid_temp, y_valid)], eval_group=[group_valid])
                
                    predictions = model.predict(x_test_temp,ntree_limit=self.model_param['n_estimators'])
               
                    new_score = ndcg_mean(group_test, y_test, predictions, 10)
                
                    if new_score > cur_score:      
                        visited[next_state.tobytes()] = new_score
                        self.beam_k_states.append(next_state)
                        self.beam_k_features.append(next_feature_subset)
                        self.beam_k_scores.append(new_score)
                
                else:
                    new_score = visited[next_state.tobytes()]
                    if new_score > cur_score:
                        self.beam_k_states.append(next_state)
                        self.beam_k_features.append(next_feature_subset)
                        self.beam_k_scores.append(new_score)
            

         #   after not updating best state for a time restart from previous best state"""
           # if(ct_not_best_step == self.num_not_best_step):
           #     ct_not_best_step = 0
           #     score = best_score
           #     state,feature_subset = copy.deepcopy(best_state),copy.deepcopy(best_feature_subset)
                
           # cur_state,cur_feature_subset = self.select_neighbor(copy.deepcopy(state),copy.deepcopy(feature_subset))
   
        self.beam_k_scores, self.beam_k_states, self.beam_k_features = self.sort_based_on_score(
                self.beam_k_scores, self.beam_k_states, self.beam_k_features)
    
        self.best_score =  self.beam_k_scores[0]
        self.best_features = self.beam_k_states[0]
        best_state = bitarray(self.best_features)
        temp = np.where(np.array(list(best_state))==True)[0]
        x_train_temp = x_train[:,temp]
        x_test_temp = x_test[:,temp]
        x_valid_temp = x_valid[:,temp]
        model = xgb.sklearn.XGBRanker(**self.model_param)    
        model.fit(x_train_temp,y_train,group=group_train,verbose=False,
                  eval_set=[(x_valid_temp, y_valid)], eval_group=[group_valid])
                
        best_predictions = model.predict(x_test_temp,ntree_limit=self.model_param['n_estimators'])
        self.best_scores['ndcg'] =  self.best_score
        self.best_scores['map'] = mean_ap(group_test, y_test, best_predictions, 10)
       
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

        
    
    def select_neighbor(self,state,subset):
        
        if self.options['neighbor']==1:
            """swap based neighbor selection"""
     
            candidates = list(self.feature_set - set(subset))
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
            new_state = copy.deepcopy(state)
            
            j=k=0
            
            """choosing random feature pair to insert"""
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
    
        
    def sort_based_on_score(self,scores, states, feature_subset):
        
        zipped_lists_1 = zip(scores,states)
        sorted_zipped_lists_1 = sorted(zipped_lists_1, reverse=True)
    
        sorted_states = [ i for _, i in sorted_zipped_lists_1]
    
        zipped_lists_2 = zip(scores,feature_subset)
        sorted_zipped_lists_2 = sorted(zipped_lists_2, reverse=True)
    
        sorted_feature_subset = [ i for _, i in sorted_zipped_lists_2]
    
        sorted_scores = sorted(scores, reverse = True)
        
        #if len(sorted_scores) > self.beam_width:
          #  k = int(self.beam_width+5)
          #  sorted_scores = sorted_scores[:,k]
          #  sorted_states = sorted_states[:,k]
          #  sorted_feature_subset = sorted_feature_subset[:,k]
        
        return sorted_scores, sorted_states, sorted_feature_subset