import numpy as np 
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import time
from local_beam import Local_Beam_Search
from joblib import Parallel, delayed, cpu_count



def choose_options():
    print('Choose suitable options for loacl beam search: ')
    choice = dict()
    print('Available datasets:\n1.MQ2008\n2.MQ2007\n3.OHSUMED\n4.TD2004\n5.MSLR-10K')
    choice['dataset'] = int(input("your choice in number: "))
    print('Available neighbor selection strategy:\n1.Swap\n2.Insertion')
    choice['neighbor'] = int(input("your choice in number: "))
    print('Enter the beam width')
    choice['beam_width'] = int(input("your choice in number: "))
    choice['max_iter'] = int(input("Enter maximum number of iterations: "))
    
    return choice

def load_dataset(option):
    if option==1:
        datapath = "../Datasets/MQ2008/"
    elif option==2:
        datapath = "../Datasets/MQ2007/"
    elif option==3:
        datapath = "../Datasets/OSHUMED/"
    elif option==4:
        datapath = "../Datasets/TD2004/"
    elif option==5:
        datapath = "../Datasets/MSLR-10k/"

    
    print(datapath)
        
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
              'subsample':subsample}#,'tree_method':tree_method}
    
'''calculate individual step counts'''
def calc_steps(num_features,max_iter):
    mid = num_features//2
    steps = []
    for i in range(1,num_features):
        if(i<=mid):
            step = np.ceil((max_iter*i*(num_features-i)*log_tot_state(num_features, i))
                           /(mid*(num_features-mid)*log_tot_state(num_features, mid)))
            
            if(i!=mid):
                step //= 1.5
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
              y_valid,y_test,group_train,group_valid,group_test,beam_width):
    print('yass')
    feature_selection_model = Local_Beam_Search(model_param=params,beam_width = beam_width, n_steps=int(steps[feature_number-1]),k=feature_number,options = choice)
            
    feature_selection_model.fit(x_train,x_valid,x_test,y_train,y_valid,y_test,group_train,group_valid,group_test)
    # avg_ndcg_score += feature_selection_model.best_score['ndcg']
    # avg_map_score += feature_selection_model.best_score['map']
    
    return feature_selection_model.best_scores['ndcg'],\
        feature_selection_model.best_scores['map']

def main():
    """load dataset"""
    datasets = ['MQ2008','MQ2007','OHSUMED','TD2004','MSLR-10K']
    choice = choose_options()
    x_train,y_train,x_valid,y_valid,x_test,y_test,group_train,group_valid,\
    group_test = load_dataset(choice['dataset'])
    
    
    print("No. of rows in training set ",'-----', len(group_train))
    print('\n')
    
    beam_width = choice['beam_width']
    
    print('The beam width is :', beam_width)
    
    total_features = x_train.shape[1]
    
    model_name = 'lambdaMART'
    params = load_ltr_params(model_name)
    
    output_path = '../Files_lb/Loacal_beam_search_result_neighbor#'+str(choice['neighbor'])+'_beam_width#'\
            +str(choice['beam_width'])+'_dataset#'+str(choice['dataset'])\
            +'2.csv'
    open(output_path,'w').close()
    title = 'Feature_Number,Avg_NDCG@10,Std_NDCG@10,Best_NDCG@10,Avg_MAP@10,Std_MAP@10,Best_MAP@10\n'
    outf = open(output_path,'a')
    outf.write(title)
    outf.close()
    
    
    steps= calc_steps(total_features, choice['max_iter'])
    print(sum(steps))
    best_score = dict()
    repeat_num=4

    for feature_number in range(1,total_features):
        t1 = time.time()
        
        best_score = dict()
        avg_score = dict()
        std = dict()
        best_features = None
        

        all_scores = Parallel(n_jobs=cpu_count())(
            delayed(run_model)(feature_number, steps, params, choice, x_train, x_valid, x_test, y_train, y_valid, y_test, group_train, group_valid, group_test,beam_width)
            for i in range(repeat_num))
        
        # all_scores = [run_model(feature_number, steps, params, choice, x_train, x_valid, x_test, y_train, y_valid, y_test, group_train, group_valid, group_test,beam_width) for i in range(repeat_num)]
        
        t2 = time.time()

        all_ndcg_scores = [x[0] for x in all_scores]
        all_map_scores = [x[1] for x in all_scores]
        #print(all_ndcg_scores)
        

        avg_score['ndcg'] = np.mean(all_ndcg_scores)
        avg_score['map'] = np.mean(all_map_scores)
        std['ndcg'] = np.std(all_ndcg_scores)
        std['map'] = np.std(all_map_scores)
        best_score['ndcg'] = np.amax(all_ndcg_scores)
        best_score['map'] = np.amax(all_map_scores)
        
        print('----------------------------------------')
        print('For k: ',feature_number,'---------- Time require:-------- ',t2-t1)
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
