import asyncio
import gym

from HIL.agent import HIL


async def main():

    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    #env = gym("MountainCar-v0", render_mode="rgb_array")

    # hyperparameters
    discount_factor = 1
    epsilon = 0  
    min_eps = 0
    num_episodes = 2
    tame = True  

    tamer_training_timestep = 0.3

    agent = HIL(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    await agent.train(model_file_to_save='autosave')
    agent.play(n_episodes=100, render=True)
    agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())




