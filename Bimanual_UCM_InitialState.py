# Identifying actions belonging to the Uncontrolled Manifold for bimanual manipulation of a cursor. The cursor is placed at the mean position of the two hands forming a redundant system.
import numpy as np
import matplotlib.pyplot as plt

Hand_Actions = np.arange(0,5) # Each hand can perform one of the five actions : stay, right, down, left or up
coordchange_action = {0:[0,0],1:[1,0],2:[0,-1],3:[-1,0],4:[0,1]} # The coordinate change of the hand for each action

Grid = np.arange(-1,2) # size of the grid for movement of the cursor
Bimanual_Actions = [[i,j] for i in Hand_Actions for j in Hand_Actions] # pairs of action between left and right hands
States = [[i,j] for i in Grid for j in Grid] # Space of cursor states
StateAction_Value= [[i,j,0,0] for i in States for j in Bimanual_Actions] # Expected future rewards from taking an action At from state St


episodes = 1000

# Chooses an action using the epsilong greedy method using the state action values.
def choose_action(state):
    Action_Values=[]
    for V in StateAction_Value:
        if V[0]==state:
            Action_Values.append(V[1:])
    Values = [I[1] for I in Action_Values]

    e=np.random.random()
    if e > 0.2:
        Index = np.random.choice(np.flatnonzero(Values == np.max(Values)))
    else:
        Index = np.random.choice(np.arange(0,len(Values)))

    chosen_action = Action_Values[Index][0]
    return chosen_action

# Reward is 1 if the cursor state is stabilized or doesnt change
def determine_reward(state):
    if np.array_equal(state,[0,0]):
        return 1
    else:
        return -1

# Update the state action values after each episode to identify new policies for greedy behavior
def update_stateaction_values(History):
    G=0
    for i in np.arange(len(History)-3,-1,-3):
        G = 0.8 * G + History[i+2]
        for index,V in enumerate(StateAction_Value):
            if V[0]==History[i] and V[1]==History[i+1]:
                StateAction_Value[index][3] += 1
                StateAction_Value[index][2] = StateAction_Value[index][2]+(G - StateAction_Value[index][2])/StateAction_Value[index][3]
                break

for i in range(episodes):
    History = []
    Cursor_state = [0,0]

    History.append(Cursor_state)
    action=choose_action(Cursor_state)
    History.append(action)

    New_Cursor_state= Cursor_state + np.add(coordchange_action[action[0]],coordchange_action[action[1]])/2
    Reward = determine_reward(New_Cursor_state)
    History.append(Reward)

    update_stateaction_values(History)


    #print("The new cursor state is {} and the action was {}".format(Cursor_state,action))

# Plot the number of times particular action was chosen
Action_Values = []
Actions = []
for V in StateAction_Value:
    if V[0] == [0,0]:
        Action_Values.append(V[3]/10)
        Actions.append(str(V[1]))


# Plotting the number of times each action was picked out of 1000 episodes. Gives an idea about the most favorable actions for stabilizing the cursor or actions belonging to the Uncontrolled Manifold
Bar_Plot = plt.subplot()
Bar_Plot.bar(np.arange(len(Action_Values)),Action_Values,align = 'center')
Bar_Plot.set_ylabel('Action choice percentage (%)')
Bar_Plot.set_xlabel('Actions')
Bar_Plot.set_yticks(np.arange(0,50,10))
Bar_Plot.set_xticks(np.arange(len(Actions)))
Bar_Plot.set_xticklabels(Actions)

fig = plt.gcf()
fig.set_size_inches(15, 8)

plt.show()

plt.savefig("Action_Choice%_state[0,0]" + ".png", bbox_inches="tight")











