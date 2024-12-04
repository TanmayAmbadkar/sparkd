from gymnasium.envs.box2d.lunar_lander import heuristic, step_api_compatibility
import gymnasium
import imageio

def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    images = []
    while True:
        a = heuristic(env, s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if render:
            still_open = env.render()
            images.append(still_open)
            if still_open is False:
                break

        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward, images

env = gymnasium.make("LunarLander-v2", render_mode="rgb_array") 
_, images = demo_heuristic_lander(env, render=True)
imageio.mimsave(f"lander.gif", images, fps=30)