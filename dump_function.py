import numpy as np


def dump_result(top_num, depth, k, singleton_relax, singleton_strict, acc):
    with open('final_result/final_result_no_tune_top_'+str(top_num)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('fold:')
        f.write(str(k))
        f.write('\n')
        f.write('final_singleton_relax:')
        f.write(str(singleton_relax))
        f.write('\n')
        f.write('final_singleton_relax_length:')
        f.write(str((len(singleton_relax))))
        f.write('\n')
        f.write('final_singleton_strict:')
        f.write(str(singleton_strict))
        f.write('\n')
        f.write('final_singleton_strict_length:')
        f.write(str((len(singleton_strict))))
        f.write('\n')
        f.write('FIDELITYS:')
        f.write(str(acc))
        f.write('\n')
    f.close()


def dump_simplified_result(top_num, depth, rule_str_relax, rule_str_strict):
    with open('final_result/final_result_no_tune_top_'+str(top_num)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('simplified rule relax:')
        f.write(str(rule_str_relax))
        f.write('\n')
        f.write('simplified rule strict:')
        f.write(str(rule_str_strict))
        f.write('\n')
    f.close()


def dump_time(top_num, depth, tree_candiate_time, kmap_time):
    with open('final_result/final_result_no_tune_top_'+str(top_num)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('tree times:')
        f.write(str(tree_candiate_time))
        f.write('\n')
        f.write('kmap times:')
        f.write(str(kmap_time))
        f.write('\n')
    f.close()


def dump_parameter(top, depth, pid):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Top:')
        f.write(str(top))
        f.write('\n')
        f.write('PID:')
        f.write(str(pid))
        f.write('\n')
        f.write('Tree depth:')
        f.write(str(depth))
        f.write('\n')
    f.close()


def dump_fidelity_scores(top, depth, fidelity_scores, rank):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('fidelity_scores[rank]:')
        f.write(str(fidelity_scores[rank][:]))
        f.write('\n')
    f.close()


def dump_final_statistics(top, depth, k, fid_list, rules_list, avg_fix, fid_all_depth, avg_problem, problem_id, consistent_tree_len_all, tree_candiate_time_list, kmap_time_list):
    with open('final_result/Statistic_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Test data Fold:')
        f.write(str(k))
        f.write('\n')
        f.write('Top:')
        f.write(str(top))
        f.write('\n')
        f.write('fid_list:')
        f.write(str(fid_list))
        f.write('\n')
        f.write('All consistent rules:')
        f.write(str(rules_list))
        f.write('\n')
        f.write('AVG_FIDELITYS_per_fold:')
        f.write(str(avg_fix))
        f.write('\n')
        f.write('AVG_FIDELITYS_all:')
        f.write(str(fid_all_depth))
        f.write('\n')
        f.write('AVG_FIDELITYS_all_problem:')
        f.write(str(avg_problem))
        f.write('\n')
        f.write('Problem_id:')
        f.write(str(problem_id))
        f.write('\n')
        f.write('Consistent trees count all avg:')
        f.write(str(np.mean(consistent_tree_len_all)))
        f.write('\n')
        f.write('trees time all avg:')
        f.write(str(np.mean(tree_candiate_time_list)))
        f.write('\n')
        f.write('kmap time all avg:')
        f.write(str(np.mean(kmap_time_list)))
        f.write('\n')
    f.close()


def dump_nonICM_parameters(top, depth, ele, mu_N, mu_I, delta, Q, situation_num):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Singleton:')
        f.write(str(ele))
        f.write('\n')
        f.write('mu_N:')
        f.write(str(mu_N))
        f.write('  ')
        f.write('mu_I:')
        f.write(str(mu_I))
        f.write('  ')
        f.write('delta:')
        f.write(str(delta))
        f.write('  ')
        f.write('Q:')
        f.write(str(Q))
        f.write('\n')
        f.write('situation_num:')
        f.write(str(situation_num))
        f.write('\n')
    f.close()


def dump_ICM_parameters(top, depth, ele, ICM_threshold, delta, Q, situation_num):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Singleton:')
        f.write(str(ele))
        f.write('\n')
        f.write('ICM_threshold:')
        f.write(str(ICM_threshold))
        f.write('  ')
        f.write('delta:')
        f.write(str(delta))
        f.write('  ')
        f.write('Q:')
        f.write(str(Q))
        f.write('\n')
        f.write('situation_num:')
        f.write(str(situation_num))
        f.write('\n')
    f.close()


def dump_nonICM_notused_parameters(top, depth, ele, mu_N, mu_I, delta, Q, situation_num):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Not Used Singleton:')
        f.write(str(ele))
        f.write('\n')
        f.write('mu_N:')
        f.write(str(mu_N))
        f.write('  ')
        f.write('mu_I:')
        f.write(str(mu_I))
        f.write('  ')
        f.write('delta:')
        f.write(str(delta))
        f.write('  ')
        f.write('Q:')
        f.write(str(Q))
        f.write('\n')
        f.write('situation_num:')
        f.write(str(situation_num))
        f.write('\n')
    f.close()


def dump_ICM_notused_parameters(top, depth, ele, ICM_threshold, delta, Q, situation_num):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Not Used Singleton:')
        f.write(str(ele))
        f.write('\n')
        f.write('ICM_threshold:')
        f.write(str(ICM_threshold))
        f.write('  ')
        f.write('delta:')
        f.write(str(delta))
        f.write('  ')
        f.write('Q:')
        f.write(str(Q))
        f.write('\n')
        f.write('situation_num:')
        f.write(str(situation_num))
        f.write('\n')
    f.close()


def dump_final_singleton(top, depth, ele, decision_list_relax, decision_list_strict):
    with open('final_result/final_result_no_tune_top_'+str(top)+'_depth_'+str(depth)+'.txt', 'a') as f:
        f.write('Singleton relax: ')
        f.write(str(decision_list_relax[ele]))
        f.write('\n')
        f.write('Singleton strict: ')
        f.write(str(decision_list_strict[ele]))
        f.write('\n')
        f.write('\n')
    f.close()
