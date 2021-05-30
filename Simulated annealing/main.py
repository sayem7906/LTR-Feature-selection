import numpy as np 
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import time
from simulated_annealing import simulated_annealing
import time
from joblib import Parallel, delayed, cpu_count



def choose_options():
    print('Choose suitable options:')
    choice = dict()
    print('Available datasets:\n1.MQ2008\n2.MQ2007\n3.OHSUMED\n4.TD2004\n5.MSLR-10K')
    choice['dataset'] = int(input("your choice in number: "))
    print('Available neighbor selection strategy:\n1.Swap\n2.Insertion')
    choice['neighbor'] = int(input("your choice in number: "))
    print('Available cooling schedule:\n1.Geometric\n2.Logarithmic\n3.Fast Annealing')
    choice['schedule'] = int(input("your choice in number: "))
    choice['max_iter'] = int(input("Enter maximum number of iterations: "))
    if choice['schedule']==1:
        choice['cooldown_factor'] = float(input('Enter cooldown factor: '))
    else:
        choice['cooldown_factor'] = 0.9
    return choice

def load_dataset(option):
    if option==1:
        datapath = "../Datasets/MQ2008/"
    elif option==2:
        datapath = "../Datasets/MQ2007/"
    elif option==3:
        datapath = "../Datasets/OHSUMED/"
    elif option==4:
        datapath = "../Datasets/TD2004/"
    elif option==5:
        datapath = "../Datasets/MSLR-10K/"
        
    x_train,y_train = load_svmlight_file(datapath+'train_dat.txt')
    x_valid, y_valid = load_svmlight_file(datapath+"vali_dat.txt")
    x_test, y_test = load_svmlight_file(datapath+"test_dat.txt")
    
    group_train = []
    with open(datapath+"train_dat.txt.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_train.append(int(line.split("\n")[0]))
    
    group_valid = []
    with open(datapath+"vali_dat.txt.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_valid.append(int(line.split("\n")[0]))
    
    group_test = []
    # with open(datapath+"test_qids.txt", "r") as f:
    #     data = f.readlines()
    #     for line in data:
    #         group_test.append(int(line.split("\n")[0].replace('qid:', '')))
    with open(datapath+"test_dat.txt.group", "r") as f:
        data = f.readlines()
        for line in data:
            group_test.append(int(line.split("\n")[0]))
    return x_train,y_train,x_valid,y_valid,x_test,y_test,group_train,\
    group_valid,group_test
        
def load_ltr_params(model):
    if model == 'lambdaMART':
        objective  = 'rank:pairwise'
        learning_rate = 0.1
        gamma = 1
        min_child_weight =1
        n_estimators = 50
        max_leaves = 5
        subsample = 0.5
        tree_method = 'hist'
        return {'objective': objective, 'learning_rate': learning_rate,
              'gamma':gamma, 'min_child_weight': min_child_weight,
              'n_estimators': n_estimators,'max_leaves': max_leaves,
              'subsample':subsample,'tree_method':tree_method}
    
    

    
# print(ndcg(y_true, y_pred, k))
'''calculate individual step counts'''
def calc_steps(num_features,max_iter):
    mid = num_features//2
    steps = []
    for i in range(1,num_features):
        if(i<=mid):
            step = np.ceil((max_iter*i*(num_features-i)*log_tot_state(num_features, i))
                           /(mid*(num_features-mid)*log_tot_state(num_features, mid)))
            
            # if(i!=mid):
            #     step //= 1.5
            if(step<=0):
                step=1.0
            steps.append(step)
        else:
            steps.append(steps[num_features-i-1])
    return steps

def log_tot_state(n,r):
    if(r<n/2):
        r = n-r
    total = 0
    for i in range(n-r+1,n):
        total += np.log(i)
    for i in range(2,r+1):
        total -= np.log(i)
    return total


def run_model(feature_number,steps,params,choice,x_train,x_valid,x_test,y_train,
              y_valid,y_test,group_train,group_valid,group_test):
    if steps[feature_number-1]<=300:   
        feature_selection_model = simulated_annealing(model_param=params,
                                n_steps=int(steps[feature_number-1]),
                                k=feature_number,options = choice,num_acc_steps=5,
                                num_not_best_step=15)
    else:
        feature_selection_model = simulated_annealing(model_param=params,
                                n_steps=int(steps[feature_number-1]),
                                k=feature_number,options = choice,num_acc_steps=10,
                                num_not_best_step=30)
        
    feature_selection_model.fit(x_train,x_valid,x_test,y_train,y_valid,y_test,group_train,group_valid,group_test)
    # avg_ndcg_score += feature_selection_model.best_score['ndcg']
    # avg_map_score += feature_selection_model.best_score['map']
    
    return feature_selection_model.best_score['ndcg'],\
        feature_selection_model.best_score['map']


def main():
    """load dataset"""
    
    datasets = ['MQ2008','MQ2007','OHSUMED','TD2004','MSLR-10K']
    choice = choose_options()
    print(choice)
    x_train,y_train,x_valid,y_valid,x_test,y_test,group_train,group_valid,\
    group_test = load_dataset(choice['dataset'])
    
    
    # print(x_train)
    
    total_features = x_train.shape[1]
    
    model_name = 'lambdaMART'
    params = load_ltr_params(model_name)
    
    
    output_path = '../Files/'+datasets[choice['dataset']-1]+'/result_neighbor#'+str(choice['neighbor'])+'_schedule#'\
            +str(choice['schedule'])+'_dataset#'+str(choice['dataset'])\
            +'_noprog.csv'
    open(output_path,'w').close()
    title = 'Feature_Number,Avg_NDCG@10,Std_NDCG@10,Best_NDCG@10,Avg_MAP@10,\
        Std_MAP@10,Best_MAP@10\n'
    outf = open(output_path,'a')
    outf.write(title)
    outf.close()
    
    
    steps= calc_steps(total_features, choice['max_iter'])
    print(sum(steps))
    best_score = dict()
    repeat_num= 10
    for feature_number in range(1,total_features):
        
        t1 = time.time()
        
        # if feature_number<3 or feature_number>(total_features-3):
        #     steps=10
            
        # elif feature_number>((total_features-1)//3) and feature_number<=(2*(total_features-1)//3):
        #     steps = 100
        # else:
        #     steps = 50
        best_score['ndcg'] = 0
        best_score['map'] = 0
        avg_score = dict()
        std = dict()
        best_features = None
        
        '''this is to parallelize'''
        all_scores = Parallel(n_jobs=cpu_count())(
            delayed(run_model)(feature_number, steps, params, choice, x_train, x_valid, x_test, y_train, y_valid, y_test, group_train, group_valid, group_test)
            for i in range(repeat_num))
        
        '''this is without parallelization'''
        # all_scores = [run_model(feature_number, steps, params, choice, x_train, x_valid, x_test, y_train, y_valid, y_test, group_train, group_valid, group_test) for i in range(repeat_num)]
        
        # for i in range(repeat_num):
        #     # print('round number: ',i,'\n\n\n')
        #     if steps[feature_number-1]<=300:   
        #         feature_selection_model = simulated_annealing(model_param=params,
        #                                 n_steps=int(steps[feature_number-1]),
        #                                 k=feature_number,options = choice,num_acc_steps=5,
        #                                 num_not_best_step=15)
        #     else:
        #         feature_selection_model = simulated_annealing(model_param=params,
        #                                 n_steps=int(steps[feature_number-1]),
        #                                 k=feature_number,options = choice,num_acc_steps=10,
        #                                 num_not_best_step=30)
                
        #     feature_selection_model.fit(x_train,x_valid,x_test,y_train,y_valid,y_test,group_train,group_valid,group_test)
        #     # avg_ndcg_score += feature_selection_model.best_score['ndcg']
        #     # avg_map_score += feature_selection_model.best_score['map']
            
        #     all_ndcg_scores[i] = feature_selection_model.best_score['ndcg']
        #     all_map_scores[i] = feature_selection_model.best_score['map']
        
        
            
        t2 = time.time()
        all_ndcg_scores = [x[0] for x in all_scores]
        all_map_scores = [x[1] for x in all_scores]
        # print(all_ndcg_scores)
        # print(all_ndcg_scores)
        # input('wait')
        avg_score['ndcg'] = np.mean(all_ndcg_scores)
        avg_score['map'] = np.mean(all_map_scores)
        std['ndcg'] = np.std(all_ndcg_scores)
        std['map'] = np.std(all_map_scores)
        best_score['ndcg'] = np.amax(all_ndcg_scores)
        best_score['map'] = np.amax(all_map_scores)
        
        print('----------------------------------------')
        print('For k: ',feature_number,' Time require: ',t2-t1)
        print('average ndcg score: ',avg_score['ndcg'])
        print('average map score: ',avg_score['map'])
        print('Best ndcg score: ',best_score['ndcg'])
        print('Best map score: ',best_score['map'])
        # print('Best features: ',best_features)
        print('Standard deviation for ndcg: ',std['ndcg'])
        print('Standard deviation for map: ',std['map'])
        print('----------------------------------------\n')
        
        buffer_s = str(feature_number) + ',' + str(avg_score['ndcg']) + ',' + str(std['ndcg']) +\
            ',' + str(best_score['ndcg']) + ',' + str(avg_score['map']) + ',' + str(std['map']) + ',' \
            + str(best_score['map']) +','+ '\n'
        outf = open(output_path,'a')
        outf.write(buffer_s)
        outf.close()
        print('\n------file writing done------\n')
    
if __name__ == "__main__":
    main()
