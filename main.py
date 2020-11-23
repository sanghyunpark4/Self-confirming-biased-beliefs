# Self-confirming biased beliefs in learning by doing process Ver 1.0

# Importing modules
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T = 1000  # number of periods to simulate the model
sampleSize = 1000  # sample size
information = 0 # if 0, then agents get information on chosen action only (i.e., own-action dependence).
                # if 1, then agents get information on all actions (i.e., complete information).
                # if 2, then agents get information on random action (i.e., random information).

choiceRule = 2  # if 0, then agents do greedy search.
                # if 1, then agents follow soft-max rule with fixed tau.
                # if 2, then agents follow soft-max rule with endogeneous tau.
tau = 0.05   # Soft-max temperature for fixed tau case
updateRule = 0  # if 0, then agents follow a Bayesian updating.
                # if 1, then agents follow ERWA.
phi = 0.1   # degree of recency for EWRA rule

if information == 0:
    print("information is restricted to chosen action")
elif information == 1:
    print("information is available for all actions")
elif information == 2:
    print("information is restricted to random action")

if choiceRule == 0:
    if updateRule == 0:
        print("Agents follow greedy search and bayesian updating")
    elif updateRule == 1:
        print("Agents follow greedy search and ERWA")
elif choiceRule == 1:
    if updateRule == 0:
        print("Agents follow soft-max with fixed tau  and bayesian updating")
    elif updateRule == 1:
        print("Agents follow soft-max with fixed tau and ERWA")
elif choiceRule == 2:
    if updateRule == 0:
        print("Agents follow soft-max with endogenous tau and bayesian updating")
    elif updateRule == 1:
        print("Agents follow soft-max with endogenous and ERWA")

# Task environment
M = 50  # number of alternatives
noise = 0   # if noise = 0, then there is no noise in feedback. If noise > 0, then feedback is noisy.
reality = np.zeros(M)

######################################################################################################

# DEFINING RESULTS VECTORS
avg_agent_perf = np.zeros((T, sampleSize))
correct_choice = np.zeros((T, sampleSize))
biased_beliefs = np.zeros((T, sampleSize))
switching_behavior = np.zeros((T, sampleSize))

# Defining functions
def genEnvironment(M):  # Generate task environment
    r = np.random.rand(M)
    return r

def genPriors(M): # Generate random priors
    r = np.random.rand(M)
    return r


def getBest(reality):  # max action selection
    best = reality.argmax(axis=0)
    return best

def dissimilarity(l1, l2):
    diff = l1 - l2
    c = float(np.sum(np.absolute(diff), axis=None) / M)
    return c

def hardmax(attraction, t, M):  # max action selection
    maxcheck = attraction.max(axis=0)
    mincheck = attraction.min(axis=0)
    if maxcheck == mincheck:
        choice = random.randint(0, M - 1)
    else:
        choice = attraction.argmax(axis=0)
    return choice


def softmax(attraction, t, M, tau):  # softmax action selection
    prob = np.zeros((1, M))
    denom = 0
    i = 0
    while i < M:
        denom = denom + math.exp((attraction[i]) / tau)
        i = i + 1
    roulette = random.random()
    i = 0
    p = 0
    while i < M:
        prob[0][i] = math.exp(attraction[i] / tau) / denom
        p = p + prob[0][i]
        if p > roulette:
            choice = i
            return choice
            break  # stops computing probability of action selection as soon as cumulative probability exceeds roulette
        i = i + 1


# Time varying model objects
attraction = np.zeros(M)

# To keep track of c(c)ount of # times action selected for bayesian updating
count = np.ones(M)

# SIMULTAION IS RUN HERE

for a in range(sampleSize):
    reality = genEnvironment(M)
    attraction = genPriors(M)
    count = np.ones(M)
    bestchoice = getBest(reality)
    pchoice = -1

    for t in range(T):
        if choiceRule == 0:
            choice = hardmax(attraction, t, M)
        elif choiceRule == 1:
            choice = softmax(attraction, t, M, tau)
        elif choiceRule == 2:
            if t < 2:
                tau = 1
            else:
                tau = 1 - avg_agent_perf[t-1, a]

            if tau <= 0.01:
                choice = hardmax(attraction, t, M)
            else:
                choice = softmax(attraction, t, M, tau)


        payoff = reality[choice] + noise*(0.5-random.random())

        avg_agent_perf[t][a] = payoff
        if choice == bestchoice:
            correct_choice[t][a] = 1
        biased_beliefs[t][a] = dissimilarity(reality, attraction)
        if choice != pchoice:
            switching_behavior[t][a] = 1
        pchoice = choice

        if information == 0:
            count[choice] += 1
            if updateRule == 0:
                attraction[choice] = (count[choice]-1)/count[choice]*attraction[choice] + 1/count[choice]*payoff
            elif updateRule == 1:
                attraction[choice] = (1-phi) * attraction[choice] + phi*payoff
        elif information == 1:
            for k in range(M):
                count[k] += 1
                payoff = reality[k] + noise*(0.5-random.random())
                if updateRule == 0:
                    attraction[k] = (count[k] - 1) / count[k] * attraction[k] + 1 / count[k] * payoff
                elif updateRule == 1:
                    attraction[k] = (1 - phi) * attraction[k] + phi * payoff
        elif information == 2:
            k = random.randint(0, M - 1)
            count[k] += 1
            payoff = reality[k] + noise * (0.5 - random.random())
            if updateRule == 0:
                attraction[k] = (count[k] - 1) / count[k] * attraction[k] + 1 / count[k] * payoff
            elif updateRule == 1:
                attraction[k] = (1 - phi) * attraction[k] + phi * payoff

result_org = np.zeros((T, 5))

for t in range(T):  # Compiling final output
    result_org[t, 0] = t
    result_org[t, 1] = float(np.sum(avg_agent_perf[t, :])) / sampleSize
    result_org[t, 2] = float(np.sum(correct_choice[t, :])) / sampleSize
    result_org[t, 3] = float(np.sum(biased_beliefs[t, :])) / sampleSize
    result_org[t, 4] = float(np.sum(switching_behavior[t, :])) / sampleSize

# WRITING RESULTS TO CSV FILE
filename = ("SCBB" + " (information condition = " + str(information) +"; choice rule = "+ str(choiceRule)+"; updating rule =  "+ str(updateRule)+ ').csv')
with open(filename, 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(
        ['Period', 'Performance', 'Proportion of the optimal choice', 'Distance between beliefs and reality', 'Switching behavior'])
    for values in result_org:
        thewriter.writerow(values)
    f.close()

##PRINTING END RUN RESULTS
print("Final performance " + str(result_org[T - 1, 1]))
print("Final proportion of the optimal choice " + str(result_org[T - 1, 2]))
print("Final distance between beliefs and reality " + str(result_org[T - 1, 3]))

#
##GRAPHICAL OUTPUT
plt.style.use('ggplot')  # Setting the plotting style
fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
ax1 = plt.subplot2grid((6, 5), (4, 1), colspan=3)

ax1.plot(result_org[:, 0], result_org[:, 1], color='black', linestyle='--', label='Average Performance')
ax1.plot(result_org[:, 0], result_org[:, 2], color='blue', linestyle='--', label='Proportion of the optimal choice')
ax1.plot(result_org[:, 0], result_org[:, 3], color='black', linestyle='--', label='Distance between beliefs and reality')
ax1.plot(result_org[:, 0], result_org[:, 4], color='black', linestyle='--', label='Switching behavior')

ax1.set_xlabel('t', fontsize=12)
ax1.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
# print 'Computation time:', TIMEDIFFERENCE
