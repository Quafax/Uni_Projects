import os
import subprocess


class Gridworld:
    def __init__(self):
        self.grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "S", ".", "D", "#", ".", "#"],
            ["#", ".", ".", ".", "M", ".", "#"],
            ["#", ".", "#", "#", "#", ".", "#"],
            ["#", ".", "#", "D", "#", ".", "#"],
            ["#", ".", ".", ".", "#", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#"]
        ]
        # #:=Wall, S:=Start, D:=Death, M:=Mud, G:=Goal
        self.start_position =(1,1)
        self.position = self.start_position
        self.terminated = False

    def reset(self):
        self.position = self.start_position
        self.terminated = False
        return self.position

    def step(self, action):
        self.position = self.trans(self.position, action)
        reward = self.rew(self.position)

        row, col = self.position
        self.terminated = self.grid[row][col] in ("G", "D")

        return self.position, reward, self.terminated
    
    def render(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if(row, col)==self.position:
                    print("A", end=" ")
                else:
                    print(self.grid[row][col], end=" ")

            print()

    def save_states(self):
        states = []
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if self.grid[row][col] != "#":
                    states.append((row, col))
        return states
    
    def play_gridworld(self):
        self.reset()

        total_return = 0
        gamma = 1
        t = 0

        directions = {"w": 0, "a": 1, "s": 2, "d": 3}

        while not self.terminated:
            subprocess.run(
                ["cmd", "/c", "cls"] if os.name == "nt" else ["clear"]
            )

            self.render()
            print("Return:", total_return)

            direction = input("Enter Action (w/a/s/d): ").strip()
            if direction not in directions:
                continue

            _, reward, termination = self.step(directions[direction])
            total_return += gamma**t * reward
            t += 1

            if termination:
                break

        self.render()
        print("Final Return:", total_return)


    def trans(self, state, action):
        move_dic = {
            0: (-1, 0),   # hoch
            1: (0, -1),   # links
            2: (1, 0),    # runter
            3: (0, 1)     # rechts
        }

        if action not in move_dic:
            raise ValueError("action needs to be 0,1,2 or 3")

        #get state
        row, col = state
        delta_row, delta_col = move_dic[action]
        #calc new position
        new_row = row + delta_row
        new_col = col + delta_col
        #return new position if in grid and no wall, else return old state
        if (0 <= new_row < len(self.grid)
                and 0 <= new_col < len(self.grid[0])):
            if self.grid[new_row][new_col] != "#":
                return (new_row, new_col)

        return state

    def rew(self, next_state):
        row, col = next_state
        cell = self.grid[row][col]

        rewards = {
            ".": 0,
            "S": 0,
            "G": 1,
            "D": -1,
            "M": -0.1
        }

        if cell not in rewards:
            raise ValueError("Cell kein gültiges symbol")

        return rewards[cell]

    def value_iteration(self,gamma=1.0,theta=1e-6):
        states = self.save_states()
        values = {state: 0.0 for state in states}
        policy = {}
        while True:
            #änderung der values, damit man aufhörn kann wenn die null ist
            delta = 0.0
            #gehe alle states durch
            for state in states:
                row, col = state
                #save old value for delta calculation
                old_value = values[state]

                action_values = []
                if self.grid[row][col] in ("G", "D"):
                    continue # because these are terminal states
                #gehe alle moves für den state durch
                for action in range(4):
                    #der nächste state nach dem move
                    next_state = self.trans(state, action)
                    #der reward für den nächsten state
                    reward = self.rew(next_state)
                    #calc value for action
                    value = reward + gamma * values[next_state]
                    #list all action values (current sate)
                    action_values.append(value)
                #beste action value für den state in values dict speichern
                values[state] = max(action_values)
                #calc policy
                policy[state] = action_values.index(max(action_values))
                #änderung des values
                change = abs(old_value - values[state])
                #update delta to max change 
                delta = max(delta, change)
            if delta < theta:
                break
        return values, policy

    def policy_extraction(self, values, gamma=1.0):
        policy={}
        for state in values:
            row, col = state
            if self.grid[row][col] in ("G", "D"):
                continue # because again terminal states
            action_values = []
            for action in range(4):
                next_state = self.trans(state, action)
                reward = self.rew(next_state)
                value = reward + gamma * values[next_state]
                action_values.append(value)
            #search best action
            best_action = action_values.index(max(action_values))
            policy[state] = best_action
        return policy
    
if __name__ == "__main__":
    env=Gridworld()
    env.play_gridworld()
    #save_states = env.save_states()
    #print("States saved:", save_states, "Total states:", len(save_states))

    values,policy = env.value_iteration(gamma=0.9, theta=0.00001)
    for state, value in values.items():
        print(f"{state}: {value:.4f}")
    print("\nPolicy:")
    for state, value in policy.items():
        print(f"{state}: {value:.4f}")