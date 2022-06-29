import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    mode = args['mode']
    split_round = args['split_round']
    data = args['data']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, mode, args['data'])
    if mode in ['joint', 'alone']:
        script_name = [['{}_recsys_{}.py'.format(run, mode)]]
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['base', 'mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre'], [mode]]]
            user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['base', 'mf', 'mlp', 'nmf', 'ae'],
                                 ['0'], ['random-8'], [mode]]]
                item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)
            else:
                item_controls = []
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode in ['mdr']:
        script_name = [['{}_recsys_{}.py'.format(run, mode)]]
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf'],
                             ['0'], ['genre'], [mode]]]
            user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf'],
                                 ['0'], ['random-8'], [mode]]]
                item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)
            else:
                item_controls = []
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'assist':
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'],
                             ['constant-0.1', 'constant-0.3', 'constant-1', 'optim-0.1'], ['constant'], ['1']]]
            user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                                 ['0'], ['random-8'], ['assist'],
                                 ['constant-0.1', 'constant-0.3', 'constant-1', 'optim-0.1'], ['constant'], ['1']]]
                item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)
            else:
                item_controls = []
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'info':
        if data in ['ML100K', 'ML1M', 'Douban']:
            script_name = [['{}_recsys_assist.py'.format(run)]]
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['1'], ['genre'], ['assist'], ['constant-0.3'], ['constant'], ['1']]]
            assist_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            controls = assist_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'match':
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            script_name = [['{}_recsys_alone.py'.format(run)]]
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['base', 'ae'],
                             ['0'], ['genre'], ['alone'], ['none'],
                             ['none'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
            alone_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['base', 'ae'],
                                 ['0'], ['random-8'], ['alone'], ['none'],
                             ['none'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
                alone_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)

            else:
                alone_item_controls = []
            script_name = [['{}_recsys_assist.py'.format(run)]]
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'], ['constant-0.3'],
                             ['constant'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
            assist_user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                                 ['0'], ['random-8'], ['assist'], ['constant-0.3'],
                                 ['constant'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
                assist_item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)
            else:
                assist_item_controls = []
            controls = alone_user_controls + alone_item_controls + assist_user_controls + assist_item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'pl':
        script_name = [['{}_recsys_assist.py'.format(run)]]
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'], ['constant-0.3'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            user_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                                 ['0'], ['random-8'], ['assist'], ['constant-0.3'],
                                 ['constant'], ['1'], ['dp-10', 'ip-10']]]
                item_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                              control_name)
            else:
                item_controls = []
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} ' \
                '--num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open('./{}_{}.sh'.format(filename, k), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open('./{}_{}.sh'.format(filename, k), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
