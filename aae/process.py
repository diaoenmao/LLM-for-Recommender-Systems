import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(data, mode):
    if mode in ['joint', 'alone']:
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['base', 'mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['genre'], [mode]]]
            user_controls = make_controls(control_name)
            control_name = [[[data], ['item'], ['explicit', 'implicit'], ['base', 'mf', 'mlp', 'nmf', 'ae'],
                             ['0'], ['random-8'], [mode]]]
            item_controls = make_controls(control_name)
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'assist':
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'],
                             ['constant-0.1', 'constant-0.3', 'constant-1', 'optim-0.1'], ['constant'], ['1']]]
            user_controls = make_controls(control_name)
            control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['random-8'], ['assist'],
                             ['constant-0.1', 'constant-0.3', 'constant-1', 'optim-0.1'], ['constant'], ['1']]]
            item_controls = make_controls(control_name)
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'match':
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'], ['constant-0.3'],
                             ['constant'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
            user_controls = make_controls(control_name)
            control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['random-8'], ['assist'], ['constant-0.3'],
                             ['constant'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']]]
            item_controls = make_controls(control_name)
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'info':
        if data in ['ML100K', 'ML1M', 'Douban']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'], ['1'], ['genre'], ['joint']]]
            joint_controls = make_controls(control_name)
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'], ['1'], ['genre'], ['alone']]]
            alone_controls = make_controls(control_name)
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['1'], ['genre'], ['assist'], ['constant-0.3'],
                             ['constant'], ['1']]]
            assist_controls = make_controls(control_name)
            controls = joint_controls + alone_controls + assist_controls
        else:
            raise ValueError('Not valid data')
    elif mode == 'pl':
        if data in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban', 'Amazon']:
            control_name = [[[data], ['user'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['genre'], ['assist'], ['constant-0.3'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            user_controls = make_controls(control_name)
            control_name = [[[data], ['item'], ['explicit', 'implicit'], ['ae'],
                             ['0'], ['random-8'], ['assist'], ['constant-0.3'],
                             ['constant'], ['1'], ['dp-10', 'ip-10']]]
            item_controls = make_controls(control_name)
            controls = user_controls + item_controls
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    write = True
    # data = ['ML100K', 'ML1M', 'ML10M', 'Douban', 'Amazon']
    data = ['ML100K', 'ML1M']
    # mode = ['joint', 'alone', 'assist']
    mode = ['joint', 'alone', 'assist']
    controls = []
    for data_ in data:
        for mode_ in mode:
            controls += make_control_list(data_, mode_)
    processed_result_exp, processed_result_history, processed_result_each = process_result(controls)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    save(processed_result_each, os.path.join(result_path, 'processed_result_each.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extracted_processed_result_each = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    extract_processed_result(extracted_processed_result_each, processed_result_each, [])
    df_exp = make_df_result(extracted_processed_result_exp, 'exp', write)
    df_history = make_df_result(extracted_processed_result_history, 'history', write)
    df_each = make_df_result(extracted_processed_result_each, 'each', write)
    make_vis_lc(df_exp, df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history, processed_result_each = {}, {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history, processed_result_each)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    summarize_result(processed_result_each)
    return processed_result_exp, processed_result_history, processed_result_each


def extract_result(control, model_tag, processed_result_exp, processed_result_history, processed_result_each):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            if 'assist' in model_tag:
                for k in base_result['logger']['test'].history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_each[metric_name] = {'each': [None for _ in range(num_experiments)]}
                    if metric_name in ['NDCG']:
                        test = max(base_result['logger']['test'].history[k])
                        history = base_result['logger']['test'].history[k]
                        test_each = base_result['logger']['test_each'].history[k]
                        test_each = np.amax(np.array(test_each).reshape(11, -1), axis=0).tolist()
                    else:
                        test = min(base_result['logger']['test'].history[k])
                        history = base_result['logger']['test'].history[k]
                        test_each = base_result['logger']['test_each'].history[k]
                        test_each = np.amin(np.array(test_each).reshape(11, -1), axis=0).tolist()
                    processed_result_exp[metric_name]['exp'][exp_idx] = test
                    processed_result_history[metric_name]['history'][exp_idx] = history
                    processed_result_each[metric_name]['each'][exp_idx] = test_each
            else:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_each[metric_name] = {'each': [None for _ in range(num_experiments)]}
                    processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger']['test'].mean[k]
                    processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['train'].history[
                        k]
                    processed_result_each[metric_name]['each'][exp_idx] = base_result['logger']['test_each'].history[k]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
            processed_result_each[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]], processed_result_each[control[1]])
    return


def summarize_result(processed_result):
    for pivot in list(processed_result.keys()):
        if pivot == 'exp':
            processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
            processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
            processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
            processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
            processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
            processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
            processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
            processed_result[pivot] = processed_result[pivot].tolist()
        elif pivot in ['history', 'each']:
            processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
            processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
            processed_result['std'] = np.std(processed_result[pivot], axis=0)
            processed_result['max'] = np.max(processed_result[pivot], axis=0)
            processed_result['min'] = np.min(processed_result[pivot], axis=0)
            processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
            processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
            processed_result[pivot] = processed_result[pivot].tolist()
        else:
            for k, v in processed_result.items():
                summarize_result(v)
            return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result or 'each' in processed_result:
        mode_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if mode_name not in extracted_processed_result:
            extracted_processed_result[mode_name] = defaultdict()
        extracted_processed_result[mode_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[mode_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df_result(extracted_processed_result, mode_name, write):
    df = defaultdict(list)
    for exp_name in extracted_processed_result:
        control = exp_name.split('_')
        for metric_name in extracted_processed_result[exp_name]:
            df_name = '_'.join([*control[:3], *control[4:6], metric_name])
            index_name = '_'.join([control[3], *control[6:]])
            df[df_name].append(pd.DataFrame(data=[extracted_processed_result[exp_name][metric_name]],
                                            index=[index_name]))
    if write:
        startrow = 0
        writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode_name), engine='xlsxwriter')
        for df_name in df:
            concat_df = pd.concat(df[df_name])
            if len(concat_df) > 1:
                df[df_name] = concat_df
                df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
                writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
                startrow = startrow + len(df[df_name].index) + 3
        writer.save()
    else:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
    return df


def make_vis_lc(df_exp, df_history):
    control_dict = {'Joint': 'Joint', 'Alone': 'Alone', 'constant-0.1_constant': 'MTAL ($\eta_k=0.1$)',
                    'constant-0.3_constant': 'MTAL ($\eta_k=0.3$)', 'constant-1_constant': 'MTAL ($\eta_k=1.0$)',
                    'optim-0.1_constant': 'MTAL (Optimize $\eta_k$)'}
    color_dict = {'Joint': 'black', 'Alone': 'gray', 'constant-0.1_constant': 'red',
                  'constant-0.3_constant': 'orange', 'constant-1_constant': 'blue',
                  'optim-0.1_constant': 'lightblue'}
    linestyle_dict = {'Joint': '-', 'Alone': ':', 'constant-0.1_constant': '--',
                      'constant-0.3_constant': '-.', 'constant-1_constant': (0, (1, 5)),
                      'optim-0.1_constant': (0, (5, 1))}
    marker_dict = {'Joint': 'X', 'Alone': 'x', 'constant-0.1_constant': 'D',
                   'constant-0.3_constant': 'd', 'constant-1_constant': 'o',
                   'optim-0.1_constant': 's'}
    label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        data_name, data_mode, target_mode, info, data_split_mode, metric_name, stat = df_name_list
        valid_mask = stat == 'mean'
        if valid_mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            joint = {}
            alone = {}
            for (index, row) in df_exp[df_name].iterrows():
                index_list = index.split('_')
                if 'joint' in index_list:
                    joint_ = row.to_numpy()
                    joint[index] = joint_[~np.isnan(joint_)]
                if 'alone' in index_list:
                    alone_ = row.to_numpy()
                    alone[index] = alone_[~np.isnan(alone_)]
            assist = {}
            for (index, row) in df_history[df_name].iterrows():
                index_list = index.split('_')
                if 'assist' in index_list and len(index_list) == 5:
                    assist_ = row.to_numpy()
                    assist[index] = assist_[~np.isnan(assist_)]
            joint_std = {}
            alone_std = {}
            for (index, row) in df_exp[df_name_std].iterrows():
                index_list = index.split('_')
                if 'joint' in index_list:
                    joint_ = row.to_numpy()
                    joint_std[index] = joint_[~np.isnan(joint_)]
                if 'alone' in index_list:
                    alone_ = row.to_numpy()
                    alone_std[index] = alone_[~np.isnan(alone_)]
            assist_std = {}
            for (index, row) in df_history[df_name_std].iterrows():
                index_list = index.split('_')
                if 'assist' in index_list and len(index_list) == 5:
                    assist_ = row.to_numpy()
                    assist_std[index] = assist_[~np.isnan(assist_)]
            joint_values = np.array(list(joint.values())).reshape(-1)
            joint_std_values = np.array(list(joint_std.values())).reshape(-1)
            alone_values = np.array(list(alone.values())).reshape(-1)
            alone_std_values = np.array(list(alone_std.values())).reshape(-1)
            if metric_name in ['NDCG']:
                joint_best_idx = np.argmax(joint_values)
                alone_best_idx = np.argmax(alone_values)
            else:
                joint_best_idx = np.argmin(joint_values)
                alone_best_idx = np.argmin(alone_values)
            x = np.arange(11)
            joint = joint_values[joint_best_idx]
            joint_std = joint_std_values[joint_best_idx]
            joint = np.full(x.shape, joint)
            joint_std = np.full(x.shape, joint_std)
            alone = alone_values[alone_best_idx]
            alone_std = alone_std_values[alone_best_idx]
            alone = np.full(x.shape, alone)
            alone_std = np.full(x.shape, alone_std)
            fig_name = '_'.join([*df_name_list[:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            control = 'Joint'
            ax_1.errorbar(x, joint, yerr=joint_std, color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'Alone'
            ax_1.errorbar(x, alone, yerr=alone_std, color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            for assist_mode in assist:
                assist_ = assist[assist_mode]
                assist_mode_list = assist_mode.split('_')
                control = '_'.join([*assist_mode_list[2:4]])
                ax_1.errorbar(x, assist_, yerr=joint_std, color=color_dict[control], linestyle=linestyle_dict[control],
                              label=control_dict[control], marker=marker_dict[control])
            ax_1.set_xticks(x)
            ax_1.set_xlabel('Assistance Rounds', fontsize=fontsize['label'])
            ax_1.set_ylabel(metric_name, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'lc', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
