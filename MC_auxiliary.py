import traceback
import time
from lib.constants import DOOM_GAMES
try:
    import cupy as cp
    from chainer.backends import cuda
except Exception as e:
    None
import numpy as np

import chainer.functions as F

from lib.utils import log, pre_process_image_tensor, post_process_image_tensor

import gym

try:
    from lib.env_wrappers import ViZDoomWrapper
except Exception as e:
    None

from scipy.misc import imresize
from lib.data import ModelDataset

import os

ID = "MC_auxiliary"

initial_z_t = None


def action(args, W_c, b_c, z_t, h_t, c_t, gpu):
    if args.weights_type == 1:
        input = F.concat((z_t, h_t), axis=0).data
        action = F.tanh(W_c.dot(input) + b_c).data
    elif args.weights_type == 2:
        input = F.concat((z_t, h_t, c_t), axis=0).data
        dot = W_c.dot(input)
        if gpu is not None:
            dot = cp.asarray(dot)
        else:
            dot = np.asarray(dot)
        output = F.tanh(dot).data
        if output == 1.:
            output = 0.999
        action_dim = args.action_dim + 1
        action_range = 2 / action_dim
        action = [0. for i in range(action_dim)]
        start = -1.
        for i in range(action_dim):
            if start <= output and output <= (start + action_range):
                action[i] = 1.
                break
            start += action_range
        mid = action_dim // 2  # reserve action[mid] for no action
        action = action[0:mid] + action[mid + 1:action_dim]
    if gpu is not None:
        action = cp.asarray(action).astype(cp.float32)
    else:
        action = np.asarray(action).astype(np.float32)
    return action


def transform_to_weights(args, parameters):
    if args.weights_type == 1:
        W_c = parameters[0:args.action_dim * (args.z_dim + args.hidden_dim)].reshape(args.action_dim,
                                                                                     args.z_dim + args.hidden_dim)
        b_c = parameters[args.action_dim * (args.z_dim + args.hidden_dim):]
    elif args.weights_type == 2:
        W_c = parameters
        b_c = None
    return W_c, b_c


def rollout(rollout_arg_tuple):
    try:
        global initial_z_t
        generation, mutation_idx, trial, args, vision, model, gpu, W_c, b_c, max_timesteps, with_frames = rollout_arg_tuple

        random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')

        if args.in_dream:
            log(ID, "Loading random rollouts for initial frames for dream training")
            initial_z_t = ModelDataset(dir=random_rollouts_dir,
                                       load_batch_size=args.initial_z_size,
                                       verbose=False)

        # The same starting seed gets passed in multiprocessing, need to reset it for each process:
        np.random.seed()

        if not with_frames:
            log(ID, ">>> Starting generation #" + str(generation) + ", mutation #" + str(
                mutation_idx + 1) + ", trial #" + str(trial + 1))
        else:
            frames_array = []
        start_time = time.time()

        model.reset_state()


        if args.in_dream:
            z_t, _, _, _ = initial_z_t[np.random.randint(len(initial_z_t))]
            z_t = z_t[0]
            if gpu is not None:
                z_t = cuda.to_gpu(z_t)
            if with_frames:
                observation = vision.decode(z_t).data
                if gpu is not None:
                    observation = cp.asnumpy(observation)
                observation = post_process_image_tensor(observation)[0]
            else:
                # free up precious GPU memory:
                if gpu is not None:
                    vision.to_cpu()
                vision = None
            if args.initial_z_noise > 0.:
                if gpu is not None:
                    z_t += cp.random.normal(0., args.initial_z_noise, z_t.shape).astype(cp.float32)
                else:
                    z_t += np.random.normal(0., args.initial_z_noise, z_t.shape).astype(np.float32)
        else:
            if args.game in DOOM_GAMES:
                env = ViZDoomWrapper(args.game)
            else:
                env = gym.make(args.game)
            observation = env.reset()
        if with_frames:
            frames_array.append(observation)

        if gpu is not None:
            h_t = cp.zeros(args.hidden_dim).astype(cp.float32)
            c_t = cp.zeros(args.hidden_dim).astype(cp.float32)
        else:
            h_t = np.zeros(args.hidden_dim).astype(np.float32)
            c_t = np.zeros(args.hidden_dim).astype(np.float32)

        done = False
        cumulative_reward = 0
        t = 0
        while not done:
            if not args.in_dream:
                observation = imresize(observation, (args.frame_resize, args.frame_resize))
                observation = pre_process_image_tensor(np.expand_dims(observation, 0))

                if gpu is not None:
                    observation = cuda.to_gpu(observation)
                z_t = vision.encode(observation, return_z=True).data[0]

            a_t = action(args, W_c, b_c, z_t, h_t, c_t, gpu)

            if args.in_dream:
                z_t, done = model(z_t, a_t, temperature=args.temperature)
                done = done.data[0]
                if with_frames:
                    observation = post_process_image_tensor(vision.decode(z_t).data)[0]
                reward = 1
                if done >= args.done_threshold:
                    done = True
                else:
                    done = False
            else:
                observation, reward, done, _ = env.step(a_t if gpu is None else cp.asnumpy(a_t))
                model(z_t, a_t, temperature=args.temperature)
            if with_frames:
                frames_array.append(observation)

            cumulative_reward += reward

            h_t = model.get_h().data[0]
            c_t = model.get_c().data[0]

            t += 1
            if max_timesteps is not None and t == max_timesteps:
                break
            elif args.in_dream and t == args.dream_max_len:
                log(ID,
                    ">>> generation #{}, mutation #{}, trial #{}: maximum length of {} timesteps reached in dream!"
                    .format(generation, str(mutation_idx + 1), str(trial + 1), t))
                break

        if not args.in_dream:
            env.close()

        if not with_frames:
            log(ID,
                ">>> Finished generation #{}, mutation #{}, trial #{} in {} timesteps in {:.2f}s with cumulative reward {:.2f}"
                .format(generation, str(mutation_idx + 1), str(trial + 1), t, (time.time() - start_time),
                        cumulative_reward))
            return cumulative_reward
        else:
            frames_array = np.asarray(frames_array)
            if args.game in DOOM_GAMES and not args.in_dream:
                frames_array = post_process_image_tensor(frames_array)
            return cumulative_reward, np.asarray(frames_array)
    except Exception:
        print(traceback.format_exc())
        return 0.
