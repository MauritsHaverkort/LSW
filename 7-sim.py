import numpy as np
import random as rand
from matplotlib import pyplot as plt

mean_friends=10
std_friends=5
amount_agents=2000
max_steps=6
amount_to_check=100


class Agent:
    def __init__(self, x,y, mean_friends):
        self.x=x
        self.y=y
        self.N = int(rand.gauss(mu=mean_friends,sigma=std_friends))
        self.full=False

        self.neighbours = []

def give_distance(a1,a2):
    return np.sqrt((a1.x-a2.x)**2+(a1.y-a2.y)**2)

def give_random_agent(agents):
    return agents[int(rand.random()*len(agents))]

def check_if_connected(a1,a2, steps):
    if a2 in a1.neighbours:
        return 1
    elif steps==1:
        return 100
    else:
        temp_result=100
        for a in a1.neighbours:
            result = check_if_connected(a,a2,steps-1)+1

            if result<temp_result:
                temp_result=result
        return temp_result

def draw_agents(agents):
    plt.figure(figsize=(8, 8))
    
    # Draw agents as points
    for agent in agents:
        
        plt.scatter(agent.x, agent.y, color='blue', s=50)  # Draw agent

        # Draw lines between neighbors
        for neighbor in agent.neighbours:
            plt.plot([agent.x, neighbor.x], [agent.y, neighbor.y], 'k-', alpha=0.5)  # Line between neighbors

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Agent Network")
    plt.grid(True)
    plt.show()

def main():

    comlete_result_mean=[]
    comlete_result_std=[]

    #for mean_friends in range(5,10):
    parameter= [100,1000,10000,100000]#range(1000,10000,1000)
    for amount_agents in parameter:
        agents=[]
        for i in range(amount_agents):
            agents.append(Agent(rand.random(), rand.random(), mean_friends=mean_friends) )
        
        counter=0
        for a in agents:
            counter+=1
            if counter%100==0:
                print(counter/len(agents))
            while(len(a.neighbours)<a.N):
                
                attempts=10
                while(True):
                    attempts-=1
                    candidate=give_random_agent(agents)
                    
                    if rand.random()*0.3>give_distance(a,candidate) and len(candidate.neighbours)<candidate.N:
                        break
                    elif attempts==0:
                        break
                        
                a.neighbours.append(candidate)
                candidate.neighbours.append(a)

                #print(check_if_connected(a,candidate,3))
        
        results=[]
        print("testing connections")
        for i in range(amount_to_check):
            print(i/amount_to_check)
            results.append(check_if_connected(give_random_agent(agents), give_random_agent(agents),max_steps))
        
        #print(give_random_agent(agents).neighbours)

        cleaned_results=[]
        not_connected=0
        for r in results:
            if r!=100:
                cleaned_results.append(r)
            else:
                not_connected+=1
        
        plt.figure(1)
        plt.hist(cleaned_results, alpha=0.5, label=str(mean_friends))
        print(np.mean(cleaned_results))
        comlete_result_mean.append(np.mean(cleaned_results))
        comlete_result_std.append(np.std(cleaned_results)/np.sqrt(len(cleaned_results)))
        print(np.std(cleaned_results))
        print(not_connected)
        plt.legend()
    plt.figure(2)
    plt.xscale('log')

    plt.errorbar(parameter, comlete_result_mean, comlete_result_std)
    plt.show()
    #draw_agents(agents)

main()
