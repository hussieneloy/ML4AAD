"""
this program is there to compare racing algorithms
"""
import numpy as np
import scipy.stats
 
theta_hat = [2, 1, 3, 4, 3]
theta_1 = [3, 1, 5, 1, 5]
theta_2 = [1, 3, 5, 4, 7]
theta_3 = [1, 2, 1, 2, 9]
theta_4 = [2, 2, 3, 5, 10]
thetas = [theta_1, theta_2, theta_3, theta_4]
N_PERMUTATIONS = 10000
alpha = .05
 
def test(x, y):
    x = np.array(x)
    y = np.array(y)
    if np.all(x == y):
        return 1
    ground_truth = np.sum(x - y)
    permutations = [np.sum((x - y) * np.random.choice([1, -1],
        size=x.shape[0])) for _ in range(N_PERMUTATIONS)]
    p_value = scipy.stats.percentileofscore(a=permutations,
        score=ground_truth) / 100
    return p_value
 
def sht():
    runs = 0
    index = -1
    current_run = 0
    incumbent = theta_hat
    pointer = -1
    for theta in thetas:
        pointer += 1
        current_run += 1
        challenger_run = 0
        while True:
            challenger_run += 1
            current_arr = incumbent[:challenger_run]
            challenger_arr = theta[:challenger_run]
            runs +=1
            if test(current_arr, challenger_arr) <= alpha:
                break
            if challenger_run == current_run:
                incumbent = theta
                index = pointer
                break
    return index, runs
   
 
def aggresive():
    runs = 0
    index = -1
    current_run = 0
    incumbent = theta_hat
    pointer = -1
    for theta in thetas:
        pointer += 1
        current_run += 1
        challenger_run = 0
        while True:
            challenger_run += 1
            current_mean = np.mean(incumbent[:challenger_run])
            challenger_mean = np.mean(theta[:challenger_run])
            runs +=1
            if current_mean < challenger_mean:
                break
            if challenger_run == current_run:
                incumbent = theta
                index = pointer
                break
    return index, runs
 
 
if __name__ == '__main__':
    idx, runs = aggresive()
    print('Aggresive Racing :')
    print('Incumbent is theta_',idx + 1)
    print('Number of Runs = ', runs)
    print()
    idx, runs = sht()
    print('Statistical Hypothesis Test Racing :')
    print('Incumbent is theta_',idx + 1)
    print('Number of Runs = ', runs)
