from mazeenv import Maze
from QL import QLearningTable


def update():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.render()
            # get next action
            action = RL.choose_action(str(observation))
            # do this action
            observation_next, reward, done = env.step(action)
            # 学习
            RL.learn(str(observation),action,reward,str(observation_next))
            # 改变状态
            observation = observation_next
            if done:
                break
    
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()
    print(RL.Q_table)